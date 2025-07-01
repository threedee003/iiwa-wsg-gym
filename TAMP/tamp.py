
from abc import abstractmethod
import numpy as np

skill_id = {
      "Reach": 0,
      "Grasp": 1,
      "Release": 2,
      "Push": 3,
      "Pull":4,
      "Insert": 5,
      "Press": 6,
      "DoorLockOpen": 7,
      "DoorOpen": 8,
      "Turning": 9,
      "Twisting": 10,
      "BoxOpen": 11,
      "BoxClose": 12,
      "Swipe": 13,
      "DrawerOpen": 14,
      "DrawerClose": 15,
      "Move": 16
}





class TAMP:
      def __init__(self):
            self.subtask_holder = []
            self.num_subtasks = 0
            self.stop_holder = []
            self.num_stops = 0
            
            

      def add_stop(self, name: str, action: str = "jt_vel", joint_positions: list = None):
            if joint_positions is None:
                  joint_positions = [0.]*7
            self.stop_holder.append(dict(name=name, action=action, joint_positions=joint_positions))
            self.num_stops += 1

      
      def add_subtask(self, skill: str, object: str, pos: np.ndarray, quat: np.ndarray, target_pos: np.ndarray, target_quat:np.ndarray):
            self.subtask_holder.append(dict(skill_id=skill_id[skill], object=object, pos=pos, quat=quat, target_pos=target_pos, target_quat=target_quat))
            self.num_subtasks += 1

            
      def remove_stop(self):
            if self.num_stops == 0:
                  AssertionError("There are stops to remove")
            self.stop_holder.pop(0)
            self.num_stops -= 1

      def remove_subtask(self):
            if self.num_subtasks == 0:
                  AssertionError("There are no subtasks to remove")
            self.subtask_holder.pop(0)
            self.num_stops -= 1

      def get_stop(self):
            return self.stop_holder[0]
      
      def get_subtask(self):
            return self.subtask_holder[0]


      def reset_tamp(self):
            del self.subtask_holder[:]
            del self.stop_holder[:]
            self.num_stops = 0
            self.num_subtasks = 0

      

