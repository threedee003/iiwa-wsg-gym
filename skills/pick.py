
from python.iiwa_workspace.iiwa_ws import iiwaScene
from isaacgym import gymapi
import numpy as np

class Pick(iiwaScene):
    def __init__(self,
                 scale_reward: float = 1.0
                 ):
        self.scale_reward = scale_reward
        pass
    
    #NOTE: To be used only if velocity control is used.

        
    


    def compute_reward(self):
        x = None
        y = None
        reward = 0.
        dist = np.linalg.norm(x-y)
        reaching_reward = 1 - np.tanh(10.0*dist)
        reward += reaching_reward
        grasping_reward = 0.25 if self.check_grasp() else 0.
        reward += grasping_reward
        if self.scale_reward is not None:
            reward *= self.scale_reward / 2.25

        return reward