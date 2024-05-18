# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##############################################################################
# modified based on franka_cabinet.py
# substitute with centauro robot
# Author: Rui Dai
##############################################################################

import numpy as np
import os
import torch
from gym import spaces

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_apply, quat_mul
from .base.vec_task import VecTask

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

class CentauroPick(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.lift_reward_scale = self.cfg["env"]["liftRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        num_obs = 52
        num_acts = 21

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env

        low = np.ones(num_acts) * -1.
        low[0:12] = np.zeros(12)
        low[19:21] = np.zeros(2)
        high = np.ones(num_acts) * 1.
        high[0:12] = np.zeros(12)
        high[19:21] = np.zeros(2)
        self.act_space = spaces.Box(low, high)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.centauro_default_dof_pos = to_torch([0.0, 0.0, 0.0, 
                                                  0.0, 0.0, 0.0,
                                                  0.5, 0.3, 0.3, -2.2, 0.0, -0.8, 
                                                  0.5, -0.3, -0.3, -2.2, 0.0, -0.8, 0.0,
                                                  0.0, 0.0], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.centauro_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_centauro_dofs]
        self.centauro_dof_pos = self.centauro_dof_state[..., 0]
        self.centauro_dof_vel = self.centauro_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_centauro_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self._cubeA_state = self.root_state_tensor[:, self._cubeA_id]

        self.hand_pose = self.rigid_body_states[:, self.hand_handle, :]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.centauro_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    
    @property
    def action_space(self):
        """Get the environment's action space."""
        low = np.ones(self.num_actions) * -1.
        low[0:12] = np.zeros(12)
        low[19:21] = np.zeros(2)
        high = np.ones(self.num_actions) * 1.
        high[0:12] = np.zeros(12)
        high[19:21] = np.zeros(2)
        self.act_space = spaces.Box(low, high)
        return self.act_space
    
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    # def set_viewer(self):
    #     super().set_viewer()
    #     # Main loop to keep the viewer open
    #     while not self.gym.query_viewer_has_closed(self.viewer):
    #         self.gym.step_graphics(self.sim)
    #         self.gym.draw_viewer(self.viewer, self.sim, True)
    #         self.gym.sync_frame_time(self.sim)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        centauro_asset_file = "urdf/centauro_urdf/urdf/centauro_sliding_upperbody.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            centauro_asset_file = self.cfg["env"]["asset"].get("assetFileNameCentauro", centauro_asset_file)

        # load centauro asset
        asset_options = gymapi.AssetOptions()
        # asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        centauro_asset = self.gym.load_asset(self.sim, asset_root, centauro_asset_file, asset_options)

        centauro_dof_stiffness = to_torch([10000, 10000, 10000, 10000, 10000, 400, 400, 400, 400, 400, 400, 400, 400, 
                                           400, 400, 400, 400, 400, 1e6, 400, 400], dtype=torch.float, device=self.device)
        centauro_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
                                         80, 80, 80, 80, 80, 1e2, 80, 80], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.5, 0.5, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)
        
        # Create cubeA asset
        self.cubeA_size = 0.050
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        self.num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        self.num_centauro_dofs = self.gym.get_asset_dof_count(centauro_asset)

        print("num centauro bodies: ", self.num_centauro_bodies)
        print("num centauro dofs: ", self.num_centauro_dofs)

        # set centauro dof properties
        centauro_dof_props = self.gym.get_asset_dof_properties(centauro_asset)
        self.centauro_dof_lower_limits = []
        self.centauro_dof_upper_limits = []
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

        self.centauro_dof_lower_limits = to_torch(self.centauro_dof_lower_limits, device=self.device)
        self.centauro_dof_upper_limits = to_torch(self.centauro_dof_upper_limits, device=self.device)
        self.centauro_dof_speed_scales = torch.ones_like(self.centauro_dof_lower_limits)
        # self.centauro_dof_speed_scales[[7]] = 0.1
        # centauro_dof_props['effort'][3] = 20000

        # define start pose for centauro
        centauro_start_pose = gymapi.Transform()
        centauro_start_pose.p = gymapi.Vec3(0.7, 0.0, 1)
        centauro_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        # self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(1.0, 0.0, 1.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        num_centauro_shapes = self.gym.get_asset_rigid_shape_count(centauro_asset)
        max_agg_bodies = num_centauro_bodies + 3 # 1 for table, table stand, cube
        max_agg_shapes = num_centauro_shapes + 3

        self.centauros = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            centauro_actor = self.gym.create_actor(env_ptr, centauro_asset, centauro_start_pose, "centauro", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, centauro_actor, centauro_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.centauros.append(centauro_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, centauro_actor, "arm2_6")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, centauro_actor, "dagana_2_top_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, centauro_actor, "dagana_2_bottom_link")

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "arm2_6")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "dagana_2_top_link")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "dagana_2_bottom_link")
        finger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "dagana_2_tcp")

        cubeA_body_handle=self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
        
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)
        finger_pose = self.gym.get_rigid_transform(self.envs[0], finger)

        # finger_pose = gymapi.Transform()
        # finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        # finger_pose.r = lfinger_pose.r
        # finger_pose.p = finger_pose.p
        # finger_pose.r = finger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        centauro_local_grasp_pose = hand_pose_inv * finger_pose
        centauro_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.centauro_local_grasp_pos = to_torch([centauro_local_grasp_pose.p.x, centauro_local_grasp_pose.p.y,
                                                centauro_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.centauro_local_grasp_rot = to_torch([centauro_local_grasp_pose.r.x, centauro_local_grasp_pose.r.y,
                                                centauro_local_grasp_pose.r.z, centauro_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))


        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.cube_inward_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.cube_up_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))

        self.centauro_grasp_pos = torch.zeros_like(self.centauro_local_grasp_pos)
        self.centauro_grasp_rot = torch.zeros_like(self.centauro_local_grasp_rot)
        self.centauro_grasp_rot[..., -1] = 1  # xyzw
        self.centauro_lfinger_pos = torch.zeros_like(self.centauro_local_grasp_pos)
        self.centauro_rfinger_pos = torch.zeros_like(self.centauro_local_grasp_pos)
        self.centauro_lfinger_rot = torch.zeros_like(self.centauro_local_grasp_rot)
        self.centauro_rfinger_rot = torch.zeros_like(self.centauro_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_centauro_reward(
            self.reset_buf, self.progress_buf, self.actions, self._table_surface_pos[2],
            self.centauro_grasp_pos, self.centauro_grasp_rot, self.centauro_lfinger_pos, self.centauro_rfinger_pos,
            self.gripper_forward_axis, self.gripper_up_axis,
            self.cubeA_pos, self.cubeA_quat, self.cube_inward_axis, self.cube_up_axis,
            self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, 
            self.finger_dist_reward_scale, self.lift_reward_scale, self.action_penalty_scale, 
            self.num_envs, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        # self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        # print("net_cf:", self.net_cf[-1])

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.cubeA_quat=self._cubeA_state[:, 3:7]
        self.cubeA_pos=self._cubeA_state[:, :3]

        self.centauro_grasp_rot[:], self.centauro_grasp_pos[:] = tf_combine(
        hand_rot, hand_pos, self.centauro_local_grasp_rot, self.centauro_local_grasp_pos)

        self.centauro_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.centauro_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.centauro_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.centauro_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.centauro_lfinger_pos = self.centauro_lfinger_pos + quat_apply(self.centauro_lfinger_rot, to_torch([[0, 0, 1]] * self.num_envs, device=self.device) * 0.15) + \
                                    quat_apply(self.centauro_lfinger_rot, to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.02)
        self.centauro_rfinger_pos = self.centauro_rfinger_pos + quat_apply(self.centauro_rfinger_rot, to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.1)

        dof_pos_scaled = (2.0 * (self.centauro_dof_pos - self.centauro_dof_lower_limits)
                          / (self.centauro_dof_upper_limits - self.centauro_dof_lower_limits) - 1.0)
        
        to_target = self.cubeA_pos - self.centauro_grasp_pos

        self.obs_buf = torch.cat((dof_pos_scaled, self.centauro_dof_vel * self.dof_vel_scale, to_target,
                                  self.cubeA_pos, self.cubeA_quat), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)
        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

        # reset centauro
        pos = tensor_clamp(
            self.centauro_default_dof_pos.unsqueeze(0) + 
            0.25 * (torch.rand((len(env_ids), self.num_centauro_dofs), device=self.device) - 0.5),
            self.centauro_dof_lower_limits, self.centauro_dof_upper_limits)
        self.centauro_dof_pos[env_ids, :] = pos
        self.centauro_dof_vel[env_ids, :] = torch.zeros_like(self.centauro_dof_vel[env_ids])
        self.centauro_dof_targets[env_ids, :self.num_centauro_dofs] = pos

        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.centauro_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self.global_indices[env_ids, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=False):
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
            cube_heights = torch.ones_like(self.hand_pose[:, 0]) * self.cubeA_size
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

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
        # print("actions:", actions[0])
        self.actions = actions.clone().to(self.device)
        targets = self.centauro_dof_targets[:, :self.num_centauro_dofs] + self.centauro_dof_speed_scales * self.dt * self.actions * self.action_scale
        # targets = to_torch([0.5, 0.0, 0.0, 
        #                     0.0, 0.0, 0.0,
        #                     0.5, 0.3, 0.3, -2.2, 0.0, -0.8, 
        #                     0.5, -0.3, -0.3, -2.2, 0.0, -0.8, 0.0,
        #                     0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.centauro_dof_targets[:, :self.num_centauro_dofs] = tensor_clamp(
            targets, self.centauro_dof_lower_limits, self.centauro_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.centauro_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        # print("obs:", self.obs_buf[0])
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.centauro_grasp_pos[i] + quat_apply(self.centauro_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.centauro_grasp_pos[i] + quat_apply(self.centauro_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.centauro_grasp_pos[i] + quat_apply(self.centauro_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.centauro_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.cubeA_pos[i] + quat_apply(self.cubeA_quat[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeA_pos[i] + quat_apply(self.cubeA_quat[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeA_pos[i] + quat_apply(self.cubeA_quat[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeA_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # px = (self.centauro_lfinger_pos[i] + quat_apply(self.centauro_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.centauro_lfinger_pos[i] + quat_apply(self.centauro_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.centauro_lfinger_pos[i] + quat_apply(self.centauro_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.centauro_lfinger_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # px = (self.centauro_rfinger_pos[i] + quat_apply(self.centauro_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.centauro_rfinger_pos[i] + quat_apply(self.centauro_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.centauro_rfinger_pos[i] + quat_apply(self.centauro_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.centauro_rfinger_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_centauro_reward(
    reset_buf, progress_buf, actions, table_height,
    centauro_grasp_pos, centauro_grasp_rot, centauro_lfinger_pos, centauro_rfinger_pos, 
    gripper_forward_axis, gripper_up_axis, 
    cube_grasp_pos, cube_grasp_rot, cube_inward_axis, cube_up_axis, 
    dist_reward_scale, rot_reward_scale, around_handle_reward_scale, 
    finger_dist_reward_scale, lift_reward_scale, action_penalty_scale, 
    num_envs, max_episode_length
):
    # distance from hand to the drawer
    d = torch.norm(centauro_grasp_pos - cube_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(centauro_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(cube_grasp_rot, cube_inward_axis)
    axis3 = tf_vector(centauro_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(cube_grasp_rot, cube_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if right finger is right to the cube handle and left to the lef
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(centauro_lfinger_pos[:, 0] < cube_grasp_pos[:, 0],
                                       torch.where(centauro_rfinger_pos[:, 0] > cube_grasp_pos[:, 0],
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the cube
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(centauro_lfinger_pos[:, 0] - cube_grasp_pos[:, 0])
    rfinger_dist = torch.abs(centauro_rfinger_pos[:, 0] - cube_grasp_pos[:, 0])
    finger_dist_reward = torch.where(centauro_lfinger_pos[:, 0] < cube_grasp_pos[:, 0],
                                     torch.where(centauro_rfinger_pos[:, 0] > cube_grasp_pos[:, 0],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # reward for lifting cubeA
    cubeA_height = cube_grasp_pos[:, 2] - table_height
    # cubeA_lifted = cubeA_height > 0.1
    lift_reward = cubeA_height

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # rewards = dist_reward_scale * dist_reward + lift_reward_scale * lift_reward - action_penalty_scale * action_penalty

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
        + around_handle_reward_scale * around_handle_reward + lift_reward_scale * lift_reward \
        + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty
    
    # bonus for opening drawer properly
    rewards = torch.where(cubeA_height > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(cubeA_height > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cubeA_height > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    reset_buf = torch.where(cubeA_height > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, centauro_local_grasp_rot, centauro_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_centauro_rot, global_centauro_pos = tf_combine(
        hand_rot, hand_pos, centauro_local_grasp_rot, centauro_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_centauro_rot, global_centauro_pos, global_drawer_rot, global_drawer_pos
