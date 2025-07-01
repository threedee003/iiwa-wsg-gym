
from python.iiwa_workspace.iiwa_ws import iiwaScene
from isaacgym import gymapi, gymutil
import numpy as np
import math
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R


class OpenDrawer(iiwaScene):
      def __init__(self,
                   scale_reward: float = 1.,
                   ):
            super(OpenDrawer, self).__init__(control_type='velocity')
            self.scale_reward = scale_reward



      def transform_object_to_tool0(self, object_quat):
            object_rot = R.from_quat(object_quat)
            flip_rot = R.from_euler('x', 180, degrees=True)
            xy_rot = R.from_euler('z', 0, degrees=True)
            new_rot = object_rot * flip_rot * xy_rot
            return new_rot.as_quat()

      def _compute_reward(self):
            raise NotImplementedError
      
      def _success(self, pos: np.ndarray, quat: np.ndarray):
            raise NotImplementedError



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

            

