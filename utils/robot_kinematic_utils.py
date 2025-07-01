from isaacgym import gymapi





def get_joint_pos_vel(gym, env, handle) -> tuple:
      dof_states = gym.get_actor_dof_states(env, handle, gymapi.STATE_ALL)
      joint_pos = [dof_state['pos'] for dof_state in dof_states]
      joint_vel = [dof_state['vel'] for dof_state in dof_states]
      return joint_pos, joint_vel

def get_joint_names(gym, env, handle) -> list:
      joint_names = gym.get_actor_dof_names(env, handle)
      return joint_names


def get_links(gym, env, handle) -> list:
      return gym.get_actor_rigid_body_names(env, handle)






'''
hasLimits

lower

upper

driveMode

stiffness

damping

velocity

effort

friction

armature
'''

# def set_dof_properties()