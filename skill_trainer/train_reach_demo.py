from python.skills.reach import Reach
from python.FKIK.fkik import FKIK
from python.sac.agent import SAC
from python.sac.rbuff import ReplayBuffer
from python.sac.sac_utils import ActionSpace

import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import yaml
from isaacgym import gymapi
from pprint import pprint 


'''

This trainer includes a pd controller to perform demonstration for reaching.


'''




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
     # np.random.seed(42)
     config = read_config("/home/bikram/Documents/isaacgym/python/configs/reach.yaml")

     mid_pos = np.array(config['env']['mid_pos'])
     quat = np.array(config['env']['traditional_quat'])
     boundaries = np.array(config['env']['offset'])
     random_pos = config['env']['randomise_target_pos']
     control_freq = config['env']['control_freq']
     random_interval = config['env']['randomise_after']


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
     obs_size = config['train']['obs_size']
     task_dir = config['train']['task_dir']
     act_sp = config['train']['action_space']

     action_space = ActionSpace(input_size=act_sp)
     agent = SAC(
          num_inputs=obs_size,
          action_space=action_space,
          gamma=gamma,
          tau=tau,
          alpha=alpha,
          policy=policy,
          target_update_interval=target_update_interval,
          automatic_entropy_tuning=automatic_entropy_tuning,
          hidden_size=hidden_size,
          learning_rate=learning_rate,
          task_dir=task_dir
     )
     agent.load_checkpoint()
     writer = SummaryWriter(task_dir+f'/runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC')
     memory = ReplayBuffer(
                    max_size=replay_buffer_size,
                    input_shape=(obs_size, ),
                    n_actions=act_sp
     )

     fixed_target = positon_sampler(mid_pos, boundaries)
     scene = Reach(
               control_freq=control_freq,
               reach_pos=fixed_target, 
               randomise_pose = random_pos, 
               initial_pos=mid_pos, 
               initial_quat=quat
     )
     gym = scene.get_gym()
     viewer = scene.get_viewer()
     sim = scene.get_sim()
     env = scene.get_env()
     
     fkik = FKIK()
     quat = scene.transform_object_to_tool0(quat)
     print(quat)
     mid_pos_robF, quat_robF = scene.to_robot_frame(mid_pos, quat)
     init_jt = [0.]*7
     init_jt = fkik.get_ik(qinit=init_jt, pos=mid_pos_robF, quat=quat_robF)
     init_jt = list(init_jt)
     init_jt.append(0.)
     init_jt.append(0.)





     gym.set_actor_dof_states(env, scene.robot_handle, init_jt, gymapi.STATE_POS)


     final_eef, quat_robF = scene.to_robot_frame(fixed_target, quat)
     final_jt = fkik.get_ik(qinit=init_jt, pos = final_eef, quat=quat_robF)
     final_jt.append(0.)
     final_jt.append(0.)


     steps = 0
     num_episode = 1
     eps_reward = 0
     updates = 0
     display(config)
     obs, _, _ = scene._get_observations()
     done = False
     #  flag = False
     
     while scene.viewer_running():

                  if num_episode == episodes:
                    
                        print("===Training Complete===")
                        break
                  if done:
                        print(f"Episode : {num_episode} | Reward : {eps_reward}")
                        writer.add_scalar('reward/train', eps_reward, num_episode)
                        scene.reset_robot(jt = init_jt)
                        
                        if random_pos and eps_reward > 590.:
                              print('===========================================')
                              print('======= Randomising position ==============')
                              print('===========================================')
                              target = positon_sampler(mid_pos, boundaries)
                              scene.update_target(pos=target, quat=np.array(config['env']['traditional_quat']))
                        steps = 0
                        eps_reward = 0
                        num_episode += 1
                  if steps > horizon:
                        print(f"Episode : {num_episode} | Reward : {eps_reward}")
                        writer.add_scalar('reward/train', eps_reward, num_episode)
                        scene.reset_robot(jt = init_jt)
                        
                        if random_pos and eps_reward > 590.:
                              print('===========================================')
                              print('======= Randomising position ==============')
                              print('===========================================')
                              target = positon_sampler(mid_pos, boundaries)
                              scene.update_target(pos=target, quat=np.array(config['env']['traditional_quat']))
                        steps = 0
                        eps_reward = 0
                        num_episode += 1

                  actions = scene.calculate_vel_cmd(jt=final_jt)

            #    if num_episode < warmup:
            #         actions = np.random.uniform(low=-1., high=1., size = (7))
            #       #   actions = scene.calculate_vel_cmd(jt=final_jt)
            #    else:
            #         actions = agent.select_action(obs)
            #         # actions = scene.calculate_vel_cmd(jt=final_jt)
            

                  if memory.can_sample(batch_size=batch_size):
                        for i in range(updates_per_step):
                              critic1_loss, critic2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                                    memory, batch_size, updates
                              )
                              writer.add_scalar('loss/critic1', critic1_loss, updates)
                              writer.add_scalar("loss/critic2", critic2_loss, updates)
                              writer.add_scalar("loss/policy", policy_loss, updates)
                              writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                              writer.add_scalar("entropy_temperature/alpha", alpha, updates)
                              updates += 1
                              
                  scene.pre_physics_step(actions=actions)
                  scene.step()
                  next_obs, rew, done = scene.post_physics_step(step=steps)
                  eps_reward += rew
                  steps += 1

                  mask = 1 if steps == horizon else float(not done)
                  memory.store_transition(state=obs, 
                                        action=actions, 
                                        reward=rew,
                                        state_=next_obs,
                                        done=mask)
                  obs = next_obs

                  if num_episode % 10 == 0 and steps == 1:
                        print('===========================================')
                        print(f"Saving Models in episode {num_episode}")
                        print('===========================================')
                        agent.save_checkpoint(path=task_dir + "/checkpoints")

     scene.__del__()
           

