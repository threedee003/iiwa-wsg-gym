from python.skills.reach import Reach
from python.FKIK.fkik import FKIK
import numpy as np
import random
import yaml
from isaacgym import gymapi
from pprint import pprint 


def display(config):
      print("=========================")
      print("Training for < Reach > skill")
      print("=== Environment Config ===")
      pprint(config['env'])
      print("\n=== Training Config ===")
      pprint(config['train'])
      print("=========================")


def positon_sampler(mid_pos, boundaries):
    sign = np.random.choice([1, -1], size=3)
    alpha = np.random.rand(3)  
    pos = mid_pos + alpha * sign * boundaries
    return pos

def read_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train():
      config = read_config("/home/bikram/Documents/isaacgym/python/configs/reach.yaml")
      mid_pos = np.array(config['env']['mid_pos'])
      quat = np.array(config['env']['traditional_quat'])
      boundaries = np.array(config['env']['offset'])
      random_pos = config['env']['randomise_target_pos']
      replay_buffer_size = config['train']['replay_buffer_size']
      episodes = config['train']['num_episodes']
      warmup = config['train']['warmup']
      batch_size = config['train']['batch_size']
      updates_per_step = config['train']['updates_per_step']
      gamma = config['train']['gamma']
      tau = config['train']['tau']
      alpha = config['train']['alpha']
      policy = config['train']['policy']
      target_update_interval = config['train']['target_update_interval']
      automatic_entropy_tuning = config['train']['automatic_entropy_tuning']
      hidden_size = config['train']['hidden_size']
      learning_rate = config['train']['learning_rate']
      horizon = config['train']['horizon']

     
      fixed_target = positon_sampler(mid_pos, boundaries)
      scene = Reach(reach_pos=fixed_target, randomise_pose = random_pos, initial_pos=mid_pos, initial_quat=quat)

      fkik = FKIK()
      quat = scene.transform_object_to_tool0(quat)
      mid_pos_robF, quat_robF = scene.to_robot_frame(mid_pos, quat)
      init_jt = [0.]*7
      init_jt = fkik.get_ik(qinit=init_jt, pos=mid_pos_robF, quat=quat_robF)

      gym = scene.get_gym()
      viewer = scene.get_viewer()
      sim = scene.get_sim()
      env = scene.get_env()
      init_jt = list(init_jt)
      init_jt.append(0.)
      init_jt.append(0.)
      gym.set_actor_dof_states(env, scene.robot_handle, init_jt, gymapi.STATE_POS)

      steps = 0
      num_episode = 0
      eps_reward = 0
      display(config)

      while not gym.query_viewer_has_closed(viewer):
            if num_episode == episodes:
                  print("===Training Complete===")
                  break
            if steps > horizon:
                  print(f"Episode : {num_episode} | Reward : {eps_reward}")
                  print()
                  scene.reset_robot(jt = init_jt)
                  if random_pos:
                       target = positon_sampler(mid_pos, boundaries)
                       scene.update_target(pos=target, quat=np.array(config['env']['traditional_quat']))
                  steps = 0
                  eps_reward = 0
                  num_episode += 1
            
            actions = np.random.random(7) /10.
            scene.pre_physics_step(actions=actions)
            t = scene.step()
            obs, rew, done = scene.post_physics_step()
            eps_reward += rew
            steps += 1

