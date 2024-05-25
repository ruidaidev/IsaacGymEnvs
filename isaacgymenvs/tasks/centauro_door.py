##############################################################################
# the framework is modified based on franka_cube_stack.py
# substitute with centauro robot and add door
# Author: Rui Dai

##############################################################################

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply, tf_vector
from isaacgymenvs.tasks.base.vec_task import VecTask

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

class CentauroDoor(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, 
                 virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.centauro_position_noise = self.cfg["env"]["centauroPositionNoise"]
        self.centauro_rotation_noise = self.cfg["env"]["centauroRotationNoise"]
        self.centauro_dof_noise = self.cfg["env"]["centauroDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
       
        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_open_scale": self.cfg["env"]["openRewardScale"],
            "r_handle_scale": self.cfg["env"]["handleRewardScale"],
            "r_rot_scale" : self.cfg["env"]["rotRewardScale"],
            "r_around_handle_scale": self.cfg["env"]["aroundHandleRewardScale"],
            "r_finger_dist_scale": self.cfg["env"]["fingerDistRewardScale"],
            "r_action_penalty_scale": self.cfg["env"]["actionPenaltyScale"]
        }

        # TODO
        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 29
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 13

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_doorHandle_state = None      # Initial state of door handle for the current env
        self._init_doorPanel_state = None       # Initial state of door Panel for the current env
        self._door_handle_state = None          # Current state of door handle for the current env
        self._door_panel_state = None           # Current state of door Panel for the current env
        self._door_id = None                    # Actor ID corresponding to cubeA for a given env

        # Tensor placeholders
        self._root_state = None                 # State of root body (n_envs, 13)
        self._dof_state = None                  # State of all joints (n_envs, n_dof)
        self._q = None                          # Joint positions (n_envs, n_dof)
        self._qd = None                         # Joint velocities (n_envs, n_dof)
        self._rigid_body_state = None           # State of all rigid bodies (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._eef_state = None                  # end effector state (at grasping point)
        self._eef_lf_state = None               # end effector state (at left fingertip)
        self._eef_rf_state = None               # end effector state (at right fingertip)
        self._arm_control = None                # Tensor buffer for controlling arm
        self._gripper_control = None            # Tensor buffer for controlling gripper
        self._pos_control = None                # Position actions
        self._effort_control = None             # Torque actions
        self._centauro_effort_limits = None     # Actuator effort limits for centauro
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        self.up_axis = "z"
        self.up_axis_idx = 2
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, 
                         graphics_device_id=graphics_device_id, headless=headless, 
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)
                
        # Centauro defaults
        self.centauro_default_dof_pos = to_torch([0.0, 0.0, 0.0, 
                                                  0.0, 0.0, 0.0,
                                                  0.5, 0.3, 0.3, -2.2, 0.0, -0.8, 
                                                  0.5, -0.3, -0.3, -2.2, 0.0, -0.8, 0.0,
                                                  0.0, 0.0], device=self.device)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        # self.sim_params.use_gpu_pipeline = False
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))   

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        centauro_asset_file = "urdf/centauro_urdf/urdf/centauro_sliding_upperbody.urdf"
        door_asset_file = "urdf/door_left.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            centauro_asset_file = self.cfg["env"]["asset"].get("assetFileNameCentauro", centauro_asset_file)

        # load centauro asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        centauro_asset = self.gym.load_asset(self.sim, asset_root, centauro_asset_file, asset_options)

        centauro_dof_stiffness = to_torch([10000, 10000, 10000, 10000, 10000, 400, 400, 400, 400, 400, 400, 400, 400, 
                                           400, 400, 400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        centauro_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
                                         80, 80, 80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        # Create door asset
        door_pos = [0.0, 0.4, 1.0]
        door_opts = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        door_opts.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        door_asset = self.gym.load_asset(self.sim, asset_root, door_asset_file, door_opts)

        self.num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        self.num_centauro_dofs = self.gym.get_asset_dof_count(centauro_asset)
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        self.num_door_dofs = self.gym.get_asset_dof_count(door_asset)

        print("num centauro bodies: ", self.num_centauro_bodies)
        print("num centauro dofs: ", self.num_centauro_dofs)
        print("num door bodies: ", self.num_door_bodies)
        print("num door dofs: ", self.num_door_dofs)

        # set centauro dof properties
        centauro_dof_props = self.gym.get_asset_dof_properties(centauro_asset)
        self.centauro_dof_lower_limits = []
        self.centauro_dof_upper_limits = []
        self._centauro_effort_limits = []
        for i in range(self.num_centauro_dofs):
            centauro_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                centauro_dof_props['stiffness'][i] = centauro_dof_stiffness[i]
                centauro_dof_props['damping'][i] = centauro_dof_damping[i]
            else:
                centauro_dof_props['stiffness'][i] = 7000.0
                centauro_dof_props['damping'][i] = 50.0

            if i == 1 or i == 3 or i == 4:
                centauro_dof_props['lower'][i] = 0.0
                centauro_dof_props['upper'][i] = 0.0
            if i == 5:
                centauro_dof_props['lower'][i] = -1.0
                centauro_dof_props['upper'][i] = 1.0
            self.centauro_dof_lower_limits.append(centauro_dof_props['lower'][i])
            self.centauro_dof_upper_limits.append(centauro_dof_props['upper'][i])
            self._centauro_effort_limits.append(centauro_dof_props['effort'][i])

        self.centauro_dof_lower_limits = to_torch(self.centauro_dof_lower_limits, device=self.device)
        self.centauro_dof_upper_limits = to_torch(self.centauro_dof_upper_limits, device=self.device)
        self._centauro_effort_limits = to_torch(self._centauro_effort_limits, device=self.device)

        # set door dof properties
        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
        for i in range(self.num_door_dofs):
            door_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            door_dof_props["stiffness"][i] = 10
            door_dof_props['damping'][i] = 10

        # Define start pose for centauro
        centauro_start_pose = gymapi.Transform()
        centauro_start_pose.p = gymapi.Vec3(1.0, 0.0, 1.0)
        centauro_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # Define start pose for table
        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(*door_pos)

        # compute aggregate size
        num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        num_centauro_shapes = self.gym.get_asset_rigid_shape_count(centauro_asset)
        num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        num_door_shapes = self.gym.get_asset_rigid_shape_count(door_asset)
        max_agg_bodies = num_centauro_bodies + num_door_bodies
        max_agg_shapes = num_centauro_shapes + num_door_shapes

        self.centauros = []
        self.doors = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: centauro should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create centauro
            # Potentially randomize start pose
            if self.centauro_position_noise > 0:
                rand_xy = self.centauro_position_noise * (-1. + np.random.rand(2) * 2.0)
                centauro_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1], 1.0)               
            if self.centauro_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.centauro_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                centauro_start_pose.r = gymapi.Quat(*new_quat)
            centauro_actor = self.gym.create_actor(env_ptr, centauro_asset, centauro_start_pose, "centauro", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, centauro_actor, centauro_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create door
            self._door_id= self.gym.create_actor(env_ptr, door_asset, door_start_pose, "door", i, 0, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.centauros.append(centauro_actor)
            self.doors.append(self._door_id)

        # Setup init state buffer
        self._init_door_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        centauro_handle = 0

        self.handles = {
            # Centauro
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, centauro_handle, "arm2_6"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, centauro_handle, "dagana_2_top_link"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, centauro_handle, "dagana_2_bottom_link"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, centauro_handle, "dagana_2_tcp"),
            # Door
            "door_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._door_id, "door_handle"),
            "door_panel": self.gym.find_actor_rigid_body_handle(self.envs[0], self._door_id, "door_panel"),
        }

        self.dof_handles = {
            # Centauro
            "x_slider": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "x_slider"),
            "y_slider": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "y_slider"),
            "z_slider": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "z_slider"),
            "j_arm2_1": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "j_arm2_1"),
            "j_arm2_2": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "j_arm2_2"),
            "j_arm2_3": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "j_arm2_3"),
            "j_arm2_4": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "j_arm2_4"),
            "j_arm2_5": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "j_arm2_5"),
            "j_arm2_6": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "j_arm2_6"),
            "dagana_2_claw_joint": self.gym.find_actor_joint_handle(env_ptr, centauro_handle, "dagana_2_claw_joint"),
        }

        # Get total DOFs
        self.num_dofs_total = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_dofs = 10

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # print(self._dof_state[0, :, :])
        self._q = self._dof_state[:, 12:19, 0]
        self._qd = self._dof_state[:, 12:19, 1]
        self._qb = self._dof_state[:, 0:6, 0]
        self._qbd = self._dof_state[:, 0:6, 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        self._door_dof_state = self._dof_state[:, -2:, 0]
        self._door_handle_state = self._rigid_body_state[:, self.handles["door_handle"], :]
        self._door_panel_state = self._rigid_body_state[:, self.handles["door_panel"], :]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs_total), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._pos_control[:, 12:18]
        self._gripper_control = self._pos_control[:, 18]
        self._base_control = self._pos_control[:, 0:6]
        self._door_control = self._pos_control[:, -2:]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        
        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, -1, 0], device=self.device).repeat((self.num_envs, 1))
        self.handle_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.handle_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        
    def _update_states(self):
        self.states.update({
            # Centauro
            "q": self._q[:, :],
            "q_gripper": self._q[:, -1],
            "qb": self._qb[:, :],
            "eef_pos": self._eef_state[:, :3] + \
                        quat_apply(self._eef_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.01),
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3] + \
                        quat_apply(self._eef_lf_state[:, 3:7], to_torch([[0, 0, 1]] * self.num_envs, device=self.device) * 0.15) + \
                        quat_apply(self._eef_lf_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.02),
            "eef_rf_pos": self._eef_rf_state[:, :3] + \
                        quat_apply(self._eef_rf_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.1),
            # Door
            "door_handle_quat": self._door_handle_state[:, 3:7],
            "door_handle_pos": self._door_handle_state[:, :3] + \
                        quat_apply(self._door_handle_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.05),
            "door_handle_angle": self._door_dof_state[:, -1].unsqueeze(-1),
            "door_panel_quat": self._door_panel_state[:, 3:7],
            "door_panel_pos": self._door_panel_state[:, :3],
            "door_panel_angle": self._door_dof_state[:, 0].unsqueeze(-1),
            "door_handle_pos_relative": self._door_handle_state[:, :3] - self._eef_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_centauro_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, 
            self.max_episode_length, self.gripper_forward_axis, self.gripper_up_axis, 
            self.handle_inward_axis, self.handle_up_axis, self.num_envs
        )

    def compute_observations(self):
        self._refresh()
        obs = ["door_handle_quat", "door_handle_pos", "eef_pos", "eef_quat", "door_panel_angle", "door_handle_angle", "q", "qb"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf
    
    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset door
        self._door_dof_state[env_ids, :] = torch.zeros_like(self._door_dof_state[env_ids])

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 21), device=self.device)
        pos = tensor_clamp(
            self.centauro_default_dof_pos.unsqueeze(0) +
            self.centauro_dof_noise * 2.0 * (reset_noise - 0.5),
            self.centauro_dof_lower_limits.unsqueeze(0), self.centauro_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        # pos[:, -3] = self.centauro_default_dof_pos[-3]

        # Reset the internal obs accordingly
        self._dof_state[env_ids, :self.num_centauro_dofs, 0] = pos
        # self._dof_state[env_ids, -1, 0] = 0.2
        self._dof_state[env_ids, :, 1] = torch.zeros_like(self._dof_state[env_ids, :, 1])
        self._q[env_ids, :] = pos[:, 12:19]
        self._qb[env_ids, :] = pos[:, 0:6]
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._qbd[env_ids, :] = torch.zeros_like(self._qbd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :self.num_centauro_dofs] = pos
        # self._pos_control[env_ids, -1] = 0.2
        self._effort_control[env_ids, :] = torch.zeros_like(self._pos_control)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, :].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper, u_base = self.actions[:, :6], self.actions[:, 6], self.actions[:, 7:]

        # Control arm (scale value first)
        # u_arm = u_arm * self.cmd_limit / self.action_scale
        arm_targets = self._q[:, :-1] + self.dt * u_arm * self.action_scale
        arm_targets = tensor_clamp(arm_targets, self.centauro_dof_lower_limits[12:18], self.centauro_dof_upper_limits[12:18])
        self._arm_control[:, :] = arm_targets

        # Control gripper
        u_fingers = self._q[:, -1] + self.dt * u_gripper * self.action_scale
        u_fingers = tensor_clamp(u_fingers, self.centauro_dof_lower_limits[18], self.centauro_dof_upper_limits[18])
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:] = u_fingers

        # Control base
        base_targets = self._qb[:, :] + self.dt * u_base * self.action_scale
        base_targets = tensor_clamp(base_targets, self.centauro_dof_lower_limits[0:6], self.centauro_dof_upper_limits[0:6])
        self._base_control[:, :] = base_targets

        # _door_panel_reset = torch.zeros_like(self._door_control)
        # _door_panel_reset[:, 0] = 0
        # _door_panel_reset[:, 1] = self.states["door_handle_angle"].squeeze()
        # _door_panel_reset[:, 1] = 0.2
        # _door_panel_remain = torch.zeros_like(self._door_control)
        # _door_panel_remain[:, 0] = self.states["door_panel_angle"].squeeze()
        # _door_panel_remain[:, 1] = self.states["door_handle_angle"].squeeze()
        # self._door_control[:, :] = torch.where(self.states["door_handle_angle"] < 0.5,
        #                                _door_panel_reset, _door_panel_remain)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            eef_lf_pos = self.states["eef_lf_pos"]
            eef_lf_rot = self._eef_lf_state[:, 3:7]
            eef_rf_pos = self.states["eef_rf_pos"]
            eef_rf_rot = self._eef_rf_state[:, 3:7]
            door_handle_pos = self.states["door_handle_pos"]
            door_handle_rot = self.states["door_handle_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, eef_lf_pos, eef_rf_pos, door_handle_pos), (eef_rot, eef_lf_rot, eef_rf_rot, door_handle_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_centauro_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length,
    gripper_forward_axis, gripper_up_axis, handle_inward_axis, handle_up_axis, num_envs
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor]

    # distance from hand to the handle
    d = torch.norm(states["door_handle_pos_relative"], dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(states["eef_quat"], gripper_forward_axis)
    axis2 = tf_vector(states["door_handle_quat"], handle_inward_axis)
    axis3 = tf_vector(states["eef_quat"], gripper_up_axis)
    axis4 = tf_vector(states["door_handle_quat"], handle_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if right finger is up to the cube handle and left to the bottom
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(states["eef_lf_pos"][:, 0] < (states["door_handle_pos"][:, 2]),
                                       torch.where(states["eef_rf_pos"][:, 0] > (states["door_handle_pos"][:, 2]),
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the handle
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(states["eef_lf_pos"][:, 0] - (states["door_handle_pos"][:, 0]))
    rfinger_dist = torch.abs(states["eef_rf_pos"][:, 0] - (states["door_handle_pos"][:, 0]))
    finger_dist_reward = torch.where(states["eef_lf_pos"][:, 0] < (states["door_handle_pos"][:, 0]),
                                     torch.where(states["eef_rf_pos"][:, 0] > (states["door_handle_pos"][:, 0]),
                                                 (0.02 - lfinger_dist) + (0.02 - rfinger_dist), finger_dist_reward), finger_dist_reward)
    # reward for rotating handle
    handle_reward = states["door_handle_angle"].squeeze() * around_handle_reward + states["door_handle_angle"].squeeze()

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # penalty on open before handle properly
    open_penalty = torch.zeros_like(rot_reward)
    open_penalty = torch.where(states["door_handle_angle"].squeeze() < 0.2,
                                       torch.where(states["door_panel_angle"].squeeze() > 0.2,
                                                   open_penalty - 100, open_penalty), open_penalty)

    rewards = reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_rot_scale"] * rot_reward \
        + reward_settings["r_around_handle_scale"] * around_handle_reward + reward_settings["r_handle_scale"] * handle_reward \
        + reward_settings["r_finger_dist_scale"] * finger_dist_reward - reward_settings["r_action_penalty_scale"] * action_penalty \
        + open_penalty
    
    # bonus for lifting properly
    rewards = torch.where(states["door_handle_angle"].squeeze() > 0.2, 
                          torch.where(states["door_panel_angle"].squeeze() > 0.01, rewards + 0.5, rewards), rewards)
    rewards = torch.where(states["door_handle_angle"].squeeze() > 0.2, 
                          torch.where(states["door_panel_angle"].squeeze() > 0.2, rewards + 1.0, rewards), rewards)
    rewards = torch.where(states["door_handle_angle"].squeeze() > 0.2, 
                          torch.where(states["door_panel_angle"].squeeze() > 0.5, rewards + 10.0, rewards), rewards)

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf