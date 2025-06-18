import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import random
from pprint import pprint
from scipy.spatial.transform import Rotation as R



axes_geom = gymutil.AxesGeometry(0.1)
ROOT = "/home/bikram/Documents/isaacgym/assets"


class iiwaScene:

      def __init__(self,
                   control_type: str = "position"
                   ):
            self.gym = gymapi.acquire_gym()
            self.sim_params = gymapi.SimParams()
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
            sim_type = gymapi.SIM_PHYSX
            compute_device_id = 0
            graphics_device_id = 0
            self.control_type = control_type
            available_control_types = ['position', 'velocity', 'torque']
            assert control_type in available_control_types, f"only available control modes are : {available_control_types}"
            

            if sim_type == gymapi.SIM_FLEX:
                  self.sim_params.substeps = 4
                  self.sim_params.flex.solver_type = 5
                  self.sim_params.flex.num_outer_iterations = 4
                  self.sim_params.flex.num_inner_iterations = 20
                  self.sim_params.flex.relaxation = 0.75
                  self.sim_params.flex.warm_start = 0.8


            elif sim_type == gymapi.SIM_PHYSX:
                  self.sim_params.substeps = 2
                  self.sim_params.physx.solver_type = 1
                  self.sim_params.physx.num_position_iterations = 25
                  self.sim_params.physx.num_velocity_iterations = 0
                  self.sim_params.physx.num_threads = 0
                  self.sim_params.physx.use_gpu = True
                  self.sim_params.physx.rest_offset = 0.001

            self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, sim_type, self.sim_params)
            self.plane_params = gymapi.PlaneParams()

            self.plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
            self.plane_params.distance = 0
            self.plane_params.static_friction = 1
            self.plane_params.dynamic_friction = 1
            self.plane_params.restitution = 0
            self.gym.add_ground(self.sim, self.plane_params)

            cabinet_urdf = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
            iiwa_urdf = "urdf/iiwa_rg2/iiwa_wsg.urdf"

            #gripper state gripper_open = 1 gripper_close = 0
            self.gripper_state = 0

            self.asset_options = gymapi.AssetOptions()
            self.asset_options.fix_base_link = False
            self.asset_options.thickness = 0.01

            cabinet_ = self.gym.load_asset(self.sim, ROOT, cabinet_urdf, self.asset_options)
            self.asset_options.fix_base_link = True
            robot_ = self.gym.load_asset(self.sim, ROOT, iiwa_urdf, self.asset_options)
            robot_props = self.gym.get_asset_dof_properties(robot_)
            robot_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            robot_props["stiffness"][7:].fill(2500.)
            robot_props["damping"][7:].fill(400.)
            robot_props['friction'][7:].fill(10.)   #default 100.

            # 1100, 300
            if self.control_type == 'position':
                  
                  robot_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
                  robot_props["stiffness"][:7].fill(700.)
                  robot_props["damping"][:7].fill(400.)
                  # robot_props['stiffness'][:7].fill(0.)
                  # robot_props['damping'][:7].fill(400.)
             

            elif self.control_type == 'velocity':
                  robot_props["driveMode"][:7].fill(gymapi.DOF_MODE_VEL)
                  robot_props["stiffness"][:7].fill(0.)
                  robot_props["damping"][:7].fill(1200.)
                  robot_props['friction'][:7].fill(1000.)

            else:
                  robot_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
                  robot_props["stiffness"][:7].fill(0.)
                  robot_props["damping"][:7].fill(0.)

                  robot_props["driveMode"][7:].fill(gymapi.DOF_MODE_EFFORT)
                  robot_props["stiffness"][7:].fill(0.)
                  robot_props["damping"][7:].fill(0.)
                  robot_props["friction"][7:].fill(1.)


            table_ = self._create_table_()
            cube_ = self._create_cube(random_positions=False)
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            spacing = 2.0
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1) 
       
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(11.88, 4.0, 0.0)
            pose.r = gymapi.Quat(0, 0, 0.7068252, 0.7073883)
            self.cabinet_handle = self.gym.create_actor(self.env, cabinet_, pose, "cabinet", 0, 0)
            
            pose.p = gymapi.Vec3(12.0, 5.0, 0.7)
            pose.r = gymapi.Quat( 0, 0, 0.7068252, 0.7073883)
            self.robot_handle = self.gym.create_actor(self.env, robot_, pose, "robot", 0, 0)
            self.table_handle = self.gym.create_actor(self.env, table_, self.table_pose, "table", 0, 0)
            self.cube_handle = self.gym.create_actor(self.env, cube_, self.cube_pose, "cube", 0, 0)
            self.gym.set_rigid_body_color(self.env, self.cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.cube_color)
            self.gym.set_actor_dof_properties(self.env, self.robot_handle, robot_props)
            self.robot_origin = np.array([12.0, 5.0, 0.7])
            self.robot_quat = np.array([0, 0, 0.7068252, 0.7073883])


      def compute_reward(self):
            raise NotImplementedError("Modified in subclass")
      
      def reach_jt_position(self, desired_jt: np.ndarray):
            if self.control_type != 'velocity':
                  raise TypeError("Controller is position control cant do velocity control")
            kp = 1.
            kd = 1
            assert self.control_type == 'velocity', f"only for velocity control"
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            joint_vel = [dof_state['vel'] for dof_state in dof_states]
            gripper_jts = np.array([0., 0.])
            desired_jt = np.concatenate((desired_jt, gripper_jts), axis = 0)
            jt_error = desired_jt-np.array(joint_pos)
            vel_cmd = (kp*jt_error - kd*np.array(joint_vel)).tolist()
            self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, vel_cmd)


      def reached_jt(self, desired_jt: np.ndarray, eps: float = None):
            if eps is None:
                  eps = 0.039
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            joint_pos.pop()
            joint_pos.pop()
            joint_pos = np.array(joint_pos)
            dist = np.linalg.norm(joint_pos-desired_jt)
            if dist >= eps:
                  return False
            else:
                  return True

      def check_gripper_status(self, status: str):
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            if status == 'open':
                  n = np.linalg.norm(joint_pos[7]+0.04 - joint_pos[8]+0.04)
                  if n <= 0.0003:
                        return True
                  else:
                        return False

            if status == 'close':
                  pass           


      def gripper_action(self, act: str):
            if act == 'open':
                  joint_pos = [0.] * 9
                  joint_pos[7] = -0.04
                  joint_pos[8] = 0.04
                  self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, joint_pos)
            if act == 'close':
                  joint_pos = [0.]*9
                  joint_pos[7] = -0.01
                  joint_pos[8] = 0.01
                  self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, joint_pos)     




      def apply_arm_action(self, action: list):
            if self.control_type == 'position':
                  action = np.array(action)
                  dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
                  joint_pos = [dof_state['pos'] for dof_state in dof_states]
                  gripper_jts = np.array([joint_pos[7], joint_pos[8]])
                  action = np.concatenate((action, gripper_jts), axis = 0).tolist()
                  self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, action)
            elif self.control_type == 'velocity':
                  self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, action)
            else:
                  raise NotImplementedError("I will not implement it.")


      def get_state(self, handle):
            dof_states = self.gym.get_actor_rigid_body_states(self.env, handle, gymapi.STATE_ALL)
            pos = np.array([dof_states['pose']['p']['x'][0], dof_states['pose']['p']['y'][0], dof_states['pose']['p']['z'][0]])
            orien = np.array([dof_states['pose']['r']['x'][0], dof_states['pose']['r']['y'][0], dof_states['pose']['r']['z'][0], dof_states['pose']['r']['w'][0]])
            return pos.astype('float64'), orien.astype('float64')
      
      def to_robot_frame(self, pos, quat):
            robot_pos_world = np.array([12.0, 5.0, 0.7])
            robot_quat_world = np.array([0.0, 0.0, 0.707, 0.707]) 
            robot_rot_world = R.from_quat(robot_quat_world)
            T_robot_world = np.eye(4)
            T_robot_world[:3, :3] = robot_rot_world.as_matrix()
            T_robot_world[:3, 3] = robot_pos_world
            T_world_robot = np.linalg.inv(T_robot_world)
            point_world_hom = np.ones(4)
            point_world_hom[:3] = pos
            point_robot_hom = T_world_robot @ point_world_hom
            point_pos_robot = point_robot_hom[:3]
            point_rot_world = R.from_quat(quat)
            point_rot_robot = robot_rot_world.inv() * point_rot_world
            point_quat_robot = point_rot_robot.as_quat()  
            return point_pos_robot, point_quat_robot


      def __del__(self):
            print("scene deleted")


      def _create_table_(self):
            table_dims = gymapi.Vec3(0.8, 1.5, 0.04)
            self.table_pose = gymapi.Transform()
            self.table_pose.p = gymapi.Vec3(12.7, 0.5 * table_dims.y + 4.3, 0.7)
            self.asset_options.thickness = 1
            self.asset_options.armature = 0.001
            table_ = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, self.asset_options)
            return table_
      
      def _create_cube(self, random_positions: bool = True):
            cube_size = 0.05
            self.cube_color = gymapi.Vec3(np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1))
            self.cube_pose = gymapi.Transform()
            if random_positions:
                  self.cube_pose.p = gymapi.Vec3(np.random.uniform(12.6, 12.8), np.random.uniform(4.3, 5.1), 1)
            else: 
                  self.cube_pose.p = gymapi.Vec3(12.7, 5.02, 1)
            self.asset_options.thickness = 0.002
            self.asset_options.fix_base_link = False
            cube_ = self.gym.create_box(self.sim, cube_size, cube_size, cube_size, self.asset_options)
            return cube_
      
      def get_viewer(self):
            return self.viewer
      
      def get_sim(self):
            return self.sim
      
      def get_gym(self):
            return self.gym
      
      def get_env(self):
            return self.env
      
      def get_robot_handle(self):
            return self.robot_handle
      
      def step(self):
            raise NotImplementedError("In child class")

      def get_joint_pos_vel(self, handle) -> tuple:
            dof_states = self.gym.get_actor_dof_states(self.env, handle, gymapi.STATE_ALL)
            joint_pos = [float(dof_state['pos']) for dof_state in dof_states]
            joint_vel = [float(dof_state['vel']) for dof_state in dof_states]
            return joint_pos, joint_vel
      

      def post_physics_step(self):
            raise NotImplementedError("In child class")

      def pre_physics_step(self):
            raise NotImplementedError("In child class")


def deg2rad(values):
      const = math.pi/ 180.
      return const*values


if __name__ == '__main__':
      inital_pos_deg = np.array([-84.53, 32.81, 3.86, -74.98, -1.14, 60.02, 126.36])
      inital_pos_rad = deg2rad(inital_pos_deg)
      num_dofs = 7
      dof_pos = [0.]* num_dofs
      # dof_pos[5] = 1.1965584

      dof_pos = np.array(inital_pos_rad)
      gripper_pos = np.array([-0.03, 0.03])
      dof_pos = np.concatenate((dof_pos, gripper_pos), axis = 0).tolist()
      print(dof_pos)

      scene = iiwaScene()
      gym = scene.get_gym()
      viewer = scene.get_viewer()
      sim = scene.get_sim()
      env = scene.get_env()
      apply = True
      while not gym.query_viewer_has_closed(viewer):
            t = gym.get_sim_time(sim)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            if t > 6 and apply:
                  print("inside if")
                  gym.set_actor_dof_position_targets(env, scene.robot_handle, dof_pos)
                  apply = False
                  print("apply false done")

      del scene