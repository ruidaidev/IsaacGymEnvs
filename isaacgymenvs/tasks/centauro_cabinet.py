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
    tf_vector, tf_combine, quat_apply
from .base.vec_task import VecTask


class CentauroCabinet(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 47
        num_acts = 21

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        low = np.ones(num_acts) * -1.
        low[6:12] = np.zeros(6)
        low[19:21] = np.zeros(2)
        high = np.ones(num_acts) * 1.
        high[6:12] = np.zeros(6)
        high[19:21] = np.zeros(2)
        self.act_space = spaces.Box(low, high)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)


        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

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
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.centauro_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    
    @property
    def action_space(self):
        """Get the environment's action space."""
        low = np.ones(self.num_actions) * -1.
        low[6:12] = np.zeros(6)
        low[19:21] = np.zeros(2)
        high = np.ones(self.num_actions) * 1.
        high[6:12] = np.zeros(6)
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
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            centauro_asset_file = self.cfg["env"]["asset"].get("assetFileNameCentauro", centauro_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

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

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        centauro_dof_stiffness = to_torch([10000, 10000, 10000, 10000, 10000, 400, 400, 400, 400, 400, 400, 400, 400, 
                                           400, 400, 400, 400, 400, 1e6, 400, 400], dtype=torch.float, device=self.device)
        centauro_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
                                         80, 80, 80, 80, 80, 1e2, 80, 80], dtype=torch.float, device=self.device)

        self.num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        self.num_centauro_dofs = self.gym.get_asset_dof_count(centauro_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num centauro bodies: ", self.num_centauro_bodies)
        print("num centauro dofs: ", self.num_centauro_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

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

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        # create prop assets
        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        centauro_start_pose = gymapi.Transform()
        centauro_start_pose.p = gymapi.Vec3(1.4, 0.0, 0.5)
        centauro_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # compute aggregate size
        num_centauro_bodies = self.gym.get_asset_rigid_body_count(centauro_asset)
        num_centauro_shapes = self.gym.get_asset_rigid_shape_count(centauro_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_centauro_bodies + num_cabinet_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_centauro_shapes + num_cabinet_shapes + self.num_props * num_prop_shapes

        self.centauros = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
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

            cabinet_pose = cabinet_start_pose
            cabinet_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cabinet_pose.p.y += self.start_position_noise * dy
            cabinet_pose.p.z += self.start_position_noise * dz
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
                drawer_pose = self.gym.get_rigid_transform(env_ptr, drawer_handle)

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = drawer_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = drawer_pose.p.y + propy
                        prop_state_pose.p.z = drawer_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.centauros.append(centauro_actor)
            self.cabinets.append(cabinet_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, centauro_actor, "arm2_6")
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, centauro_actor, "dagana_2_top_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, centauro_actor, "dagana_2_bottom_link")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "arm2_6")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "dagana_2_top_link")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "dagana_2_bottom_link")
        finger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centauros[0], "dagana_2_tcp")

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

        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                                drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                                drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.centauro_grasp_pos = torch.zeros_like(self.centauro_local_grasp_pos)
        self.centauro_grasp_rot = torch.zeros_like(self.centauro_local_grasp_rot)
        self.centauro_grasp_rot[..., -1] = 1  # xyzw
        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., -1] = 1
        self.centauro_lfinger_pos = torch.zeros_like(self.centauro_local_grasp_pos)
        self.centauro_rfinger_pos = torch.zeros_like(self.centauro_local_grasp_pos)
        self.centauro_lfinger_rot = torch.zeros_like(self.centauro_local_grasp_rot)
        self.centauro_rfinger_rot = torch.zeros_like(self.centauro_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_centauro_reward(
            self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
            self.centauro_grasp_pos, self.drawer_grasp_pos, self.centauro_grasp_rot, self.drawer_grasp_rot,
            self.centauro_lfinger_pos, self.centauro_rfinger_pos,
            self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
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
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        self.centauro_grasp_rot[:], self.centauro_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.centauro_local_grasp_rot, self.centauro_local_grasp_pos,
                                     drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
                                     )

        self.centauro_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.centauro_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.centauro_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.centauro_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.centauro_lfinger_pos = self.centauro_lfinger_pos + quat_apply(self.centauro_lfinger_rot, to_torch([[0, 0, 1]] * self.num_envs, device=self.device) * 0.15) + \
                                    quat_apply(self.centauro_lfinger_rot, to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.02)
        self.centauro_rfinger_pos = self.centauro_rfinger_pos + quat_apply(self.centauro_rfinger_rot, to_torch([[0, 1, 0]] * self.num_envs, device=self.device) * 0.1)

        dof_pos_scaled = (2.0 * (self.centauro_dof_pos - self.centauro_dof_lower_limits)
                          / (self.centauro_dof_upper_limits - self.centauro_dof_lower_limits) - 1.0)
        to_target = self.drawer_grasp_pos - self.centauro_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.centauro_dof_vel * self.dof_vel_scale, to_target,
                                  self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset centauro
        pos = tensor_clamp(
            self.centauro_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_centauro_dofs), device=self.device) - 0.5),
            self.centauro_dof_lower_limits, self.centauro_dof_upper_limits)
        self.centauro_dof_pos[env_ids, :] = pos
        self.centauro_dof_vel[env_ids, :] = torch.zeros_like(self.centauro_dof_vel[env_ids])
        self.centauro_dof_targets[env_ids, :self.num_centauro_dofs] = pos

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        # reset props
        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.centauro_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        print("actions:", actions[0])
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
        print("obs:", self.obs_buf[0])
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

                px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.drawer_grasp_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.centauro_lfinger_pos[i] + quat_apply(self.centauro_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.centauro_lfinger_pos[i] + quat_apply(self.centauro_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.centauro_lfinger_pos[i] + quat_apply(self.centauro_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.centauro_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.centauro_rfinger_pos[i] + quat_apply(self.centauro_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.centauro_rfinger_pos[i] + quat_apply(self.centauro_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.centauro_rfinger_pos[i] + quat_apply(self.centauro_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.centauro_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_centauro_reward(
    reset_buf, progress_buf, actions, cabinet_dof_pos,
    centauro_grasp_pos, drawer_grasp_pos, centauro_grasp_rot, drawer_grasp_rot,
    centauro_lfinger_pos, centauro_rfinger_pos,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(centauro_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(centauro_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    axis3 = tf_vector(centauro_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if right finger is above the drawer handle and left below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(centauro_lfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                       torch.where(centauro_rfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the drawer
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(centauro_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    rfinger_dist = torch.abs(centauro_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    finger_dist_reward = torch.where(centauro_lfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                     torch.where(centauro_rfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # how far the cabinet has been opened out
    open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
        + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
        + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty

    # bonus for opening drawer properly
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # # prevent bad style in opening drawer
    # rewards = torch.where(centauro_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    # rewards = torch.where(centauro_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    
    # reset if drawer is open or max length reached
    reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
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
