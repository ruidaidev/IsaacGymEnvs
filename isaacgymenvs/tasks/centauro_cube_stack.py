##############################################################################
# modified based on franka_cube_stack.py
# substitute with centauro robot
# Author: Rui Dai

# TODO：not test osc
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

class CentauroCubeStack(VecTask):
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
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
            "r_rot_scale" : self.cfg["env"]["rotRewardScale"],
            "r_around_handle_scale": self.cfg["env"]["aroundHandleRewardScale"],
            "r_finger_dist_scale": self.cfg["env"]["fingerDistRewardScale"],
            "r_action_penalty_scale": self.cfg["env"]["actionPenaltyScale"]
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # TODO
        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 21
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 7

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

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
        self._j_eef = None                      # Jacobian for end effector
        self._mm = None                         # Mass matrix
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
        
        # self._q = torch.zeros((self.num_envs, 7))
        # self._qd = torch.zeros((self.num_envs, 7)) 
        
        # Centauro defaults
        self.centauro_default_dof_pos = to_torch([0.0, 0.0, 0.0, 
                                                  0.0, 0.0, 0.0,
                                                  0.5, 0.3, 0.3, -2.2, 0.0, -0.8, 
                                                  0.5, -0.3, -0.3, -2.2, 0.0, -0.8, 0.0,
                                                  0.0, 0.0], device=self.device)

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._centauro_effort_limits[12:18].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
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
        drill_asset_file = "urdf/drill.urdf"

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
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        centauro_asset = self.gym.load_asset(self.sim, asset_root, centauro_asset_file, asset_options)

        centauro_dof_stiffness = to_torch([10000, 10000, 10000, 10000, 10000, 400, 400, 400, 400, 400, 400, 400, 400, 
                                           400, 400, 400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        centauro_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
                                         80, 80, 80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 0.55]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.6, 0.6, table_thickness], table_opts)

        # # Create table stand asset
        # table_stand_height = 0.1
        # table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        # table_stand_opts = gymapi.AssetOptions()
        # table_stand_opts.fix_base_link = True
        # table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # Create cubeA asset
        self.cubeA_size = 0.05
        self.cubeB_size = 0.070
        cubeA_opts = gymapi.AssetOptions()
        cubeA_opts.collapse_fixed_joints = True
        cubeA_asset = self.gym.create_box(self.sim, *([0.05, 0.05, 0.05]), cubeA_opts)
        # cubeA_asset = self.gym.load_asset(self.sim, asset_root, drill_asset_file, cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)
        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        self.num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        self.num_centauro_dofs = self.gym.get_asset_dof_count(centauro_asset)

        print("num centauro bodies: ", self.num_centauro_bodies)
        print("num centauro dofs: ", self.num_centauro_dofs)

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

            self.centauro_dof_lower_limits.append(centauro_dof_props['lower'][i])
            self.centauro_dof_upper_limits.append(centauro_dof_props['upper'][i])
            self._centauro_effort_limits.append(centauro_dof_props['effort'][i])

        self.centauro_dof_lower_limits = to_torch(self.centauro_dof_lower_limits, device=self.device)
        self.centauro_dof_upper_limits = to_torch(self.centauro_dof_upper_limits, device=self.device)
        self._centauro_effort_limits = to_torch(self._centauro_effort_limits, device=self.device)

        # Define start pose for centauro
        centauro_start_pose = gymapi.Transform()
        centauro_start_pose.p = gymapi.Vec3(-0.6, 0.3, 0.8)
        centauro_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # # Define start pose for table stand
        # table_stand_start_pose = gymapi.Transform()
        # table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        # table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)        

        # compute aggregate size
        num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        num_centauro_shapes = self.gym.get_asset_rigid_shape_count(centauro_asset)
        max_agg_bodies = num_centauro_bodies + 2     # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_centauro_shapes + 2     # 1 for table, table stand, cubeA, cubeB

        self.centauros = []
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
                centauro_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 0.8)               
            if self.centauro_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.centauro_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                centauro_start_pose.r = gymapi.Quat(*new_quat)
            centauro_actor = self.gym.create_actor(env_ptr, centauro_asset, centauro_start_pose, "centauro", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, centauro_actor, centauro_dof_props)

            # set color
            num_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
            for body_idx in range(31):
                if body_idx == 10 or body_idx == 16 or body_idx == 14 or body_idx == 19 or body_idx == 23 or body_idx == 25 or body_idx == 28 or body_idx == 9:
                    self.gym.set_rigid_body_color(env_ptr, centauro_actor, body_idx, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.116, 0.112, 0.108))
                else:
                    self.gym.set_rigid_body_color(env_ptr, centauro_actor, body_idx, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.695, 0.368, 0.086))

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            # table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
            #                                           i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            # self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            # self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            props = self.gym.get_actor_rigid_body_properties(env_ptr, self._cubeA_id)
            props[0].mass = 0.01
            self.gym.set_actor_rigid_body_properties(env_ptr, self._cubeA_id, props)

            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self._cubeA_id)
            # shape_props[0].restitution = 1  # Set high restitution for high stiffness
            shape_props[0].friction = 5.0  # Set friction (optional)
            self.gym.set_actor_rigid_shape_properties(env_ptr, self._cubeA_id, shape_props)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.centauros.append(centauro_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

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
            "pelvis": self.gym.find_actor_rigid_body_handle(env_ptr, centauro_handle, "pelvis"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "base_link"),
            # "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        self.dof_handles = {
            # Centauro
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
        self.num_dofs = 7
        # _dof_idx = list(self.dof_handles.values())
        # self.dof_idx = torch.tensor(_dof_idx)

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
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "centauro")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, centauro_handle)['j_arm2_6']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "centauro")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        # self._cubeB_state = self._root_state[:, self._cubeB_id, :]
        self.centauro_dof_speed_scales = torch.ones_like(self._q)

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            # "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs_total), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._pos_control[:, 12:18]
        self._gripper_control = self._pos_control[:, 18]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        
        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.cube_inward_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.cube_up_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        
    def _update_states(self):
        self.states.update({
            # Centauro
            "q": self._q[:, :],
            "q_gripper": self._q[:, -1],
            "eef_pos": self._eef_state[:, :3] + \
                        quat_apply(self._eef_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.01),
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3] + \
                        quat_apply(self._eef_lf_state[:, 3:7], to_torch([[0, 0, 1]] * self.num_envs, device=self.device) * 0.15) + \
                        quat_apply(self._eef_lf_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.02),
            "eef_rf_pos": self._eef_rf_state[:, :3] + \
                        quat_apply(self._eef_rf_state[:, 3:7], to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.1),
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            # "cubeB_quat": self._cubeB_state[:, 3:7],
            # "cubeB_pos": self._cubeB_state[:, :3],
            # "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
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
            self.cube_inward_axis, self.cube_up_axis, self.num_envs
        )

    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf
    
    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        # self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        # self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 21), device=self.device)
        pos = tensor_clamp(
            self.centauro_default_dof_pos.unsqueeze(0) +
            self.centauro_dof_noise * 2.0 * (reset_noise - 0.5),
            self.centauro_dof_lower_limits.unsqueeze(0), self.centauro_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        # pos[:, -3] = self.centauro_default_dof_pos[-3]

        # Reset the internal obs accordingly
        self._dof_state[env_ids, :, 0] = pos
        self._dof_state[env_ids, :, 1] = torch.zeros_like(self._dof_state[env_ids, :, 1])
        self._q[env_ids, :] = pos[:, 12:19]
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        
        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            # other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        # min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        # min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0
        # sampled_cube_state[:, 3] = 0.7071
        # sampled_cube_state[:, 6] = 0.7071

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # # Control arm (scale value first)
        # # u_arm = u_arm * self.cmd_limit / self.action_scale
        # targets = self._q[:, :-1] + self.dt * u_arm * self.action_scale
        # targets = tensor_clamp(targets, self.centauro_dof_lower_limits[12:18], self.centauro_dof_upper_limits[12:18])
        # self._arm_control[:, :] = targets

        # # Control gripper
        # u_fingers = torch.zeros_like(self._gripper_control)
        # u_fingers[:] = torch.where(u_gripper >= 0.0, self.centauro_dof_upper_limits[-3].item(), 
        #                            self.centauro_dof_lower_limits[-3].item())
        # # Write gripper command to appropriate tensor buffer
        # self._gripper_control[:] = u_fingers


        targets = self._q[:, :] + self.dt * self.actions * self.action_scale
        targets = tensor_clamp(targets, self.centauro_dof_lower_limits[12:19], self.centauro_dof_upper_limits[12:19])
        self._pos_control[:, 12:19] = targets
        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        pelivs_pose = self._rigid_body_state[0, self.handles["pelvis"], 0:7]
        pelivs_pose = pelivs_pose.cpu()
        tensor_list_pelvis = pelivs_pose.numpy().tolist()
        tensor_list_pelvis[0] = tensor_list_pelvis[0] + 0.6
        tensor_list_pelvis[2] = tensor_list_pelvis[2] - 0.3
        # tensor_list_pelvis = rotate_pose_180_deg_around_z(tensor_list_pelvis)
        with open('centauro_pick.txt', 'a') as file:
            data_dof = self._dof_state[-1, 5:, 0]
            data_dof = data_dof.cpu()
            tensor_list_dof = data_dof.numpy().tolist()
            tensor_list = tensor_list_pelvis + tensor_list_dof
            tensor_str = ' '.join(map(str, tensor_list))
            file.write(tensor_str + '\n')

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
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            # cubeB_pos = self.states["cubeB_pos"]
            # cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos), (eef_rot, cubeA_rot)):
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
    gripper_forward_axis, gripper_up_axis, cube_inward_axis, cube_up_axis, num_envs
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    # target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    # cubeB_size = states["cubeB_size"]

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    # d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    # d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    # dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
    # dist_reward = 1 - torch.tanh(10.0 * d)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    # figure_reward = torch.zeros_like(dist_reward)
    # figure_reward = torch.where(d > 0.01,
    #                             torch.where((d_lf - d_rf) > 0.1, figure_reward + 0.5, figure_reward), figure_reward)
    # dist_reward = dist_reward + figure_reward

    axis1 = tf_vector(states["eef_quat"], gripper_forward_axis)
    axis2 = tf_vector(states["cubeA_quat"], cube_inward_axis)
    axis3 = tf_vector(states["eef_quat"], gripper_up_axis)
    axis4 = tf_vector(states["cubeA_quat"], cube_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if right finger is right to the cube handle and left to the lef
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(states["eef_lf_pos"][:, 0] < (states["cubeA_pos"][:, 0] - cubeA_size / 2),
                                       torch.where(states["eef_rf_pos"][:, 0] > (states["cubeA_pos"][:, 0] + cubeA_size / 2),
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the cube
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(states["eef_lf_pos"][:, 0] - (states["cubeA_pos"][:, 0] - cubeA_size / 2))
    rfinger_dist = torch.abs(states["eef_rf_pos"][:, 0] - (states["cubeA_pos"][:, 0] + cubeA_size / 2))
    finger_dist_reward = torch.where(states["eef_lf_pos"][:, 0] < (states["cubeA_pos"][:, 0] - cubeA_size / 2),
                                     torch.where(states["eef_rf_pos"][:, 0] > (states["cubeA_pos"][:, 0] + cubeA_size / 2),
                                                 (0.1 - lfinger_dist) + (0.1 - rfinger_dist), finger_dist_reward), finger_dist_reward)
    # reward for lifting cubeA
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_lifted = (cubeA_height - cubeA_size) > 0.01
    lift_reward = cubeA_lifted

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # rewards = dist_reward_scale * dist_reward + lift_reward_scale * lift_reward - action_penalty_scale * action_penalty

    # rewards = reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward \
    #         + around_handle_reward_scale * around_handle_reward \
    #         - action_penalty_scale * action_penalty

    rewards = reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_rot_scale"] * rot_reward \
        + reward_settings["r_around_handle_scale"] * around_handle_reward + reward_settings["r_lift_scale"] * lift_reward \
        + reward_settings["r_finger_dist_scale"] * finger_dist_reward - reward_settings["r_action_penalty_scale"] * action_penalty
    
    # bonus for lifting properly
    rewards = torch.where(cubeA_height > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(cubeA_height > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cubeA_height > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
    # offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
    # offset[:, 2] = (cubeA_size + cubeB_size) / 2
    # d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
    # align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # Dist reward is maximum of dist and align reward
    # dist_reward = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
    # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
    # gripper_away_from_cubeA = (d > 0.04)
    # stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

    # Compose rewards

    # # We either provide the stack reward or the align + dist reward
    # rewards = torch.where(
    #     stack_reward,
    #     reward_settings["r_stack_scale"] * stack_reward,
    #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
    #         "r_align_scale"] * align_reward,
    # )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf