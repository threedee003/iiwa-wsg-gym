
from python.iiwa_workspace.iiwa_ws import iiwaScene
from isaacgym import gymapi, gymutil
import numpy as np
import math
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R


class Reach(iiwaScene):
      def __init__(self,
                   scale_reward: float = 1.,
                   reach_pos: np.ndarray = np.array([0., 0., 0.]),
                   reach_quat: np.ndarray = np.array([0., 0., 0., 1.]),
                   initial_pos: np.ndarray = np.array([0., 0., 0.]),
                   initial_quat: np.ndarray = np.array([0., 0., 0., 1.]),
                   randomise_pose: bool = False
                   ):
            super(Reach, self).__init__(control_type='velocity')
            self.scale_reward = scale_reward
            self.reach_pos = reach_pos
            self.reach_quat = reach_quat
            self.initial_pos = initial_pos
            self.initial_quat = initial_quat
            source_color = (1., 0., 0.)
            target_color = (0., 1., 0.)
            self.random = randomise_pose

            self.initial_quat = self.transform_object_to_tool0(self.initial_quat)
            self.reach_quat = self.transform_object_to_tool0(self.reach_quat)
            



            # making source point for viz
            attractor_props = gymapi.AttractorProperties()
            axes_geom = gymutil.AxesGeometry(0.1) 
            sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            sphere_pose = gymapi.Transform(r=sphere_rot)
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=source_color)
            attractor_props.stiffness = 0
            attractor_props.damping = 0
            attractor_props.axes = gymapi.AXIS_ALL
            attractor_props.target.p = gymapi.Vec3(initial_pos[0], initial_pos[1], initial_pos[2])
            self.source_handle = self.gym.create_rigid_body_attractor(self.env, attractor_props)
            axes_geom = gymutil.AxesGeometry(0.1)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env, attractor_props.target)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, attractor_props.target)

            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=target_color)
            attractor_props.target.p = gymapi.Vec3(reach_pos[0], reach_pos[1], reach_pos[2])
            attractor_props.target.r = gymapi.Quat(self.reach_quat[0], self.reach_quat[1], self.reach_quat[2], self.reach_quat[3])
            self.target_handle = self.gym.create_rigid_body_attractor(self.env, attractor_props)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env, attractor_props.target)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, attractor_props.target)

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            # self.initial_state = np.copy(self.gym.get_actor_rigid_body_states(self.env, self.robot_handle, gymapi.STATE_POS))
            # self.reset_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))
            # print(self.reset_state)

      def register_init_state(self):
            self.reset_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))
            print(self.reset_state)


      def update_target(self, pos: np.ndarray, quat: np.ndarray):
            assert self.random == True, "Randomise has to be enable to update target poses"
            self.reach_pos = pos
            self.reach_quat = quat
            # engg this for eef
            self.reach_quat = self.transform_object_to_tool0(self.reach_quat)

            #clear points
            self.gym.clear_lines(self.viewer)

            source_color = (1., 0., 0.)
            target_color = (0., 1., 0.)
            attractor_props = gymapi.AttractorProperties()
            axes_geom = gymutil.AxesGeometry(0.1) 
            sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            sphere_pose = gymapi.Transform(r=sphere_rot)
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=source_color)
            attractor_props.stiffness = 0
            attractor_props.damping = 0
            attractor_props.axes = gymapi.AXIS_ALL
            attractor_props.target.p = gymapi.Vec3(self.initial_pos[0], self.initial_pos[1], self.initial_pos[2])
            self.source_handle = self.gym.create_rigid_body_attractor(self.env, attractor_props)
            axes_geom = gymutil.AxesGeometry(0.1)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env, attractor_props.target)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, attractor_props.target)

            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=target_color)
            attractor_props.target.p = gymapi.Vec3(self.reach_pos[0], self.reach_pos[1], self.reach_pos[2])
            attractor_props.target.r = gymapi.Quat(self.reach_quat[0], self.reach_quat[1], self.reach_quat[2], self.reach_quat[3])
            self.target_handle = self.gym.create_rigid_body_attractor(self.env, attractor_props)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env, attractor_props.target)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, attractor_props.target)


      def transform_object_to_tool0(self, object_quat):
            object_rot = R.from_quat(object_quat)
            flip_rot = R.from_euler('x', 180, degrees=True)
            xy_rot = R.from_euler('z', 0, degrees=True)
            new_rot = object_rot * flip_rot * xy_rot
            return new_rot.as_quat()

      def _compute_reward(self):
            '''
            Reward shaping for reaching skill.

            reaching reward -> [0, 1] : Based on pos of target and current pos.
            quaternion reward -> [0, 1] : Based on target orientation and current orientation
            success reward -> {0., 0.25} : If not reached/ reached binary reward.

            '''
            eef_pose = self.gym.get_actor_rigid_body_states(self.env, self.robot_handle, gymapi.STATE_POS)[-1]
            eef_pos = np.array([eef_pose[0][0]['x'], eef_pose[0][0]['y'], eef_pose[0][0]['z']])
            eef_quat = np.array([eef_pose[0][1]['x'], eef_pose[0][1]['y'], eef_pose[0][1]['z'], eef_pose[0][1]['w']])
            reward = 0.
            dist = np.linalg.norm(eef_pos-self.reach_pos)
            quat_dot = np.abs(np.dot(eef_quat, self.reach_quat))
            quat_error = 2 * np.arccos(np.clip(quat_dot, -1.0, 1.0))  
            quat_reward = 1 - np.tanh(10. * quat_error)
            reaching_reward = 1 - np.tanh(10.0*dist)
            reward += reaching_reward
            reward += quat_reward
            if self._success(eef_pos, eef_quat):
                  reward += 0.25
            if self.scale_reward is not None:
                  reward *= self.scale_reward / 2.25
            return reward
      
      def _success(self, pos: np.ndarray, quat: np.ndarray):
            n1 = np.linalg.norm(pos-self.reach_pos)
            n2 = np.abs(np.dot(quat, self.reach_quat))
            n2 = np.arccos(np.clip(n2, -1.0, 1.0))
            if n1 <= 0.039 and n2 <= 0.04:
                  return True
            else:
                  return False



      def reset_robot(self, jt):
            self.gym.set_actor_dof_states(self.env, self.robot_handle, jt, gymapi.STATE_POS)

      def step(self):
            t = self.gym.get_sim_time(self.sim)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
            for evt in self.gym.query_viewer_action_events(self.viewer):
                  if evt.action == 'reset' and evt.value > 0:
                        self.gym.set_sim_rigid_body_states(self.sim, self.reset_state, gymapi.STATE_ALL)
                        # self.gym.set_actor_rigid_body_states(self.env, self.robot_handle, self.initial_state, gymapi.STATE_POS)
            return t




      # NOTE : gripper proprio not included for reach.
      def _get_observations(self):
            joint_pos, joint_vel = self.get_joint_pos_vel(self.robot_handle)
            joint_vel = np.array(joint_vel[:7])
            joint_pos_cos = np.cos(np.array(joint_pos[:7]))
            joint_pos_sin = np.cos(np.array(joint_pos[:7]))
            eef_pose = self.gym.get_actor_rigid_body_states(self.env, self.robot_handle, gymapi.STATE_POS)[-1]
            eef_pos = np.array([eef_pose[0][0]['x'], eef_pose[0][0]['y'], eef_pose[0][0]['z']])
            eef_quat = np.array([eef_pose[0][1]['x'], eef_pose[0][1]['y'], eef_pose[0][1]['z'], eef_pose[0][1]['w']])
            eef_pos, eef_quat = self.to_robot_frame(pos = eef_pos, quat = eef_quat)
            target_pos, target_quat = self.to_robot_frame(pos=self.reach_pos, quat=self.reach_quat)
            proprio_ = np.concatenate((joint_pos_cos, joint_pos_sin, joint_vel, eef_pos, eef_quat), axis = 0)
            obs = np.concatenate((target_pos, target_quat, proprio_), axis = 0)
            return obs, eef_pos, eef_quat



      def pre_physics_step(self, actions: np.ndarray):
            if actions is None:
                  pass
            else:
                  assert len(actions.shape) == 1, "Please give actions as a numpy list"
                  if actions.shape[0] == 7:
                        jt_pos = np.array([0., 0.])
                        actions = np.concatenate((actions, jt_pos), axis = 0)
                  self.apply_arm_action(action=actions.tolist())

      def post_physics_step(self):
            reward = self._compute_reward()
            obs, ps, q = self._get_observations()
            done = self._success(ps, q)
            return obs, reward, done
      

      def orientation_error(desired, current):
            cc = quat_conjugate(current)
            q_r = quat_mul(desired, cc)
            return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

            

