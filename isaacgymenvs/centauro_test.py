import math
from isaacgym import gymapi
from isaacgym import gymutil
import os
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine
from isaacgym import gymtorch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

# Physics engine-specific settings for friction, if available
# sim_params.physx.contact_offset = 0.01  # Adjust contact offset if necessary
# sim_params.physx.rest_offset = 0.001  # Restitution offset
# sim_params.physx.bounce_threshold_velocity = 1.0  # Velocity below which objects won't bounce

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# material_properties = gymapi.MaterialProperties(
# static_friction=0.5,  # Static friction coefficient
# dynamic_friction=0.5,  # Dynamic friction coefficient
# restitution=0.1        # Restitution coefficient (bounciness)
# )

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 15
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

up_axis = "z"
up_axis_idx = 2

# add robot urdf asset
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
# asset_file = "urdf/anymal_c/urdf/anymal.urdf"
# asset_file = "urdf/franka_description/robots/franka_panda.urdf"
assert_file = "urdf/centauro_urdf/urdf/centauro_sliding.urdf"
cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
# asset_options.flip_visual_attachments = True
asset_options.fix_base_link = True
# asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True
# asset_options.thickness = 0.001
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
# asset_options.use_mesh_materials = True

robot_asset = gym.load_asset(sim, asset_root, assert_file, asset_options)
# robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
# robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# initial root pose for centauro actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
initial_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

# Create environment
env0 = gym.create_env(sim, env_lower, env_upper, 1)
# Create the robot instance
# pose = gymapi.Transform()
robot = gym.create_actor(env0, robot_asset, initial_pose, 'robot')

# load cabinet asset
# asset_options.flip_visual_attachments = False
# asset_options.collapse_fixed_joints = True
# asset_options.disable_gravity = False
# asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
# asset_options.armature = 0.005
# cabinet_asset = gym.load_asset(sim, asset_root, cabinet_asset_file, asset_options)
# # set cabinet dof properties
# cabinet_dof_props = gym.get_asset_dof_properties(cabinet_asset)
# for i in range(len(cabinet_dof_props)):
#     cabinet_dof_props['damping'][i] = 10.0

# cabinet_start_pose = gymapi.Transform()
# cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, up_axis_idx))
# cabinet_pose = cabinet_start_pose
# cabinet_actor = gym.create_actor(env0, cabinet_asset, cabinet_pose, "cabinet")
# gym.set_actor_dof_properties(env0, cabinet_actor, cabinet_dof_props)

# Configure DOF properties
props = gym.get_actor_dof_properties(env0, robot)
for i in range(len(props)):
    props["driveMode"][i] = gymapi.DOF_MODE_POS
    props["stiffness"][i] = 5000
    props["damping"][i] = 100.0
gym.set_actor_dof_properties(env0, robot, props)
# Set DOF drive targets
torso_dof_handle = gym.find_actor_dof_handle(env0, robot, 'torso_yaw')
gym.set_dof_target_position(env0, torso_dof_handle, 0.0)
j_arm1_1_dof_handle = gym.find_actor_dof_handle(env0, robot, 'j_arm2_1')
gym.set_dof_target_position(env0, j_arm1_1_dof_handle, 0.5)
j_arm1_2_dof_handle = gym.find_actor_dof_handle(env0, robot, 'j_arm2_2')
gym.set_dof_target_position(env0, j_arm1_2_dof_handle, -0.3)
j_arm1_3_dof_handle = gym.find_actor_dof_handle(env0, robot, 'j_arm2_3')
gym.set_dof_target_position(env0, j_arm1_3_dof_handle, -0.3)
j_arm1_4_dof_handle = gym.find_actor_dof_handle(env0, robot, 'j_arm2_4')
gym.set_dof_target_position(env0, j_arm1_4_dof_handle, -2.2)
j_arm1_5_dof_handle = gym.find_actor_dof_handle(env0, robot, 'j_arm2_5')
gym.set_dof_target_position(env0, j_arm1_5_dof_handle, 0.0)
j_arm1_6_dof_handle = gym.find_actor_dof_handle(env0, robot, 'j_arm2_6')
gym.set_dof_target_position(env0, j_arm1_6_dof_handle, -0.8)
dagana_dof_handle = gym.find_actor_dof_handle(env0, robot, 'dagana_2_claw_joint')
gym.set_dof_target_position(env0, dagana_dof_handle, 0.0)

# wheel = gym.find_actor_rigid_body_handle(env0, robot, "front_wheel")
# shape_props = gym.get_actor_rigid_shape_properties(env0, robot)
# shape_props[wheel].friction = 10.
# shape_props[wheel].rolling_friction = 10.
# shape_props[wheel].torsion_friction = 10.
# gym.set_actor_rigid_shape_properties(env0, robot, shape_props)
# hand = gym.find_actor_rigid_body_handle(env0, robot, "arm2_6")
# shape_props = gym.get_actor_rigid_shape_properties(env0, hand)
# lfinger = gym.find_actor_rigid_body_handle(env0, robot, "dagana_2_top_link")
# rfinger = gym.find_actor_rigid_body_handle(env0, robot, "dagana_2_bottom_link")
# finger = gym.find_actor_rigid_body_handle(env0, robot, "dagana_2_tcp")

# hand_pose = gym.get_rigid_transform(env0, hand)
# lfinger_pose = gym.get_rigid_transform(env0, lfinger)
# rfinger_pose = gym.get_rigid_transform(env0, rfinger)
# finger_pose = gym.get_rigid_transform(env0, finger)

# drawer_handle = gym.find_actor_rigid_body_handle(env0, cabinet_actor, "drawer_top")

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)