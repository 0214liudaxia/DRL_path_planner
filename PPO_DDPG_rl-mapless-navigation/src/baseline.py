#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
import math
#import gym
import numpy as np
import tensorflow as tf

from ddpg import *
from PPO import *
from environment import Env
from pathlib import Path
import argparse
from config import *



exploration_decay_start_step = 50000
state_dim = 16
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 0.5  # rad/s

def write_to_csv(item, file_name):
    with open(file_name, 'a') as f:
        f.write("%s\n" % item)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0, help='1 for training and 0 for testing')
    parser.add_argument('--env_id', type=int, default=2, help='env name')
    parser.add_argument('--sac', type=int, default=0, help='1 for using sac')
    parser.add_argument('--visual_obs', type=int, default=0, help='1 for using image at robot observation')
    parser.add_argument('--test_env_id', type=int, default=2, help='test environment id')
    parser.add_argument('--n_scan', type=int, default=10, help='num of scan sampled from full scan')

    args = parser.parse_args()
    return args

def main():
    
    rospy.init_node('baseline')

    # get arg
    args = parse_args()
    is_training = bool(args.train)
    env_name = 'env' + str(args.env_id)
    trained_models_dir = './src/trained_models/bl-' + env_name + '-models/' if not args.visual_obs else \
            './src/trained_models/vis_obs-' + env_name + '-models/'
    #trained_models_dir = './src/trained_models/ppo'
    env = Env(is_training, args.env_id, args.test_env_id, args.visual_obs, args.n_scan)
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    past_action = np.array([0., 0.])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')
    
    ppo=False


    if(ppo==True):

        agent = Agent(dic_agent_conf, dic_path, dic_env_conf)
        figures_path = './figures_ppo/bl-' + env_name + '/' if not args.visual_obs else \
            './figures_ppo/vis_obs-' + env_name + '/'


        """for cnt_episode in range(dic_exp_conf["TRAIN_ITERATIONS"]):
            s = env.reset()
            r_sum = 0
            for cnt_step in range(dic_exp_conf["MAX_EPISODE_LENGTH"]):
                if cnt_episode > dic_exp_conf["TRAIN_ITERATIONS"] - 10:
                    env.render()

                a = agent.choose_action(s)
                s_, r, done, _ = env.step(a)

                r /= 100
                r_sum += r
                if done:
                    r = -1

                agent.store_transition(s, a, s_, r, done)
                if cnt_step % dic_agent_conf["BATCH_SIZE"] == 0 and cnt_step != 0:
                    agent.train_network()
                s = s_

                if done:
                    break

                if cnt_step % 10 == 0:
                    print("Episode:{}, step:{}, r_sum:{}".format(cnt_episode, cnt_step, r_sum))"""


    else:

        agent = DDPG(env, state_dim, action_dim, trained_models_dir)
        figures_path = './figures_ddpg/bl-' + env_name + '/' if not args.visual_obs else \
            './figures_ddpg/vis_obs-' + env_name + '/'

    if is_training:
        print('Training mode')
        # path things

        Path(trained_models_dir + 'actor').mkdir(parents=True, exist_ok=True)
        Path(trained_models_dir + 'critic').mkdir(parents=True, exist_ok=True)
        Path(figures_path).mkdir(parents=True, exist_ok=True)

        avg_reward_his = []
        total_reward = 0
        var = 1.
        ep_rets = []
        ep_ret = 0.

        while True:
            
            state = env.reset()
            one_round_step = 0
            while True:
                if ppo:
                    a=agent.choose_action(state)
                else:
                    a = agent.action(state)
                #a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)
                #a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)

                try:
                    some_object_iterator = iter(past_action)
                except TypeError as te:
                    print(past_action, 'is not iterable')

                state_, r, done, arrive = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, done)

                if arrive:
                    result = 'Success'
                else:
                    result = 'Fail'

                if time_step > 0:
                    total_reward += r
                    ep_ret += r

                if time_step % 5000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 5000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:',avg_reward_his)
                    total_reward = 0
                    print('Mean episode return over training time step: {:.2f}'.format(np.mean(ep_rets)))
                    print('Mean episode return over current 10k training time step: {:.2f}'.format(np.mean(ep_rets[-5:])))
                    write_to_csv(np.mean(ep_rets), figures_path + 'mean_ep_ret_his.csv')
                    write_to_csv(np.mean(ep_rets[-5:]), figures_path + 'mean_ep_ret_10k_his.csv')
                    write_to_csv(avg_reward, figures_path + 'avg_reward_his.csv')
                    print('---------------------------------------------------')

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    
                    var *= 0.99
                    #if var==0:
                    #   var=a

                past_action = a
                state = state_
                one_round_step += 1

                if arrive:
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    one_round_step = 0
                    if time_step > 0:
                        ep_rets.append(ep_ret)
                        ep_ret = 0.

                if done or one_round_step >= 500:
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    if time_step > 0:
                        ep_rets.append(ep_ret)
                        ep_ret = 0.
                    break

    else:
        print('Testing mode')
        total_return = 0.
        total_step = 0
        total_path_len = 0.
        arrive_cnt = 0
        robot_name='turtlebot3_burger'
        while True:
            state = env.reset()
            one_round_step = 0

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=5)
                except:
                    pass

            robot_cur_state = data.pose[data.name.index(robot_name)].position

            while True:
                a = agent.action(state)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)
                state_, r, done, arrive = env.step(a, past_action)
                total_return += r
                past_action = a
                state = state_
                one_round_step += 1
                total_step += 1

                data = None
                while data is None:
                    try:
                        data = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=5)
                    except:
                        pass

                robot_next_state = data.pose[data.name.index(robot_name)].position
                dist = math.hypot(
                        robot_cur_state.x - robot_next_state.x,
                        robot_cur_state.y - robot_next_state.y
                        )
                total_path_len += dist
                robot_cur_state = robot_next_state

                if arrive:
                    arrive_cnt += 1
                    print('Step: %3i' % one_round_step, '| Arrive!!!')
                    one_round_step = 0
                    if env.test_goals_id >= len(env.test_goals):
                        print('Finished, total return: ', total_return)
                        print('Total step: ', total_step)
                        print('Total path length: ', total_path_len)
                        print('Success rate: ', arrive_cnt / len(env.test_goals))
                        exit(0)

                if done:
                    print('Step: %3i' % one_round_step, '| Collision!!!')
                    if env.test_goals_id >= len(env.test_goals):
                        print('Finished, total return: ', total_return)
                        print('Total step: ', total_step)
                        print('Total path length: ', total_path_len)
                        print('Success rate: ', arrive_cnt / len(env.test_goals))
                        exit(0)
                    break


if __name__ == '__main__':
    dic_agent_conf = {
    "STATE_DIM": 16,
    "ACTOR_LEARNING_RATE": 0.0001,#1e-3,
    "CRITIC_LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "PATIENCE": 10,
    "NUM_LAYERS": 2,
    #"D_DENSE": 32,
    "ACTION_DIM":2,
    "ACTOR_LOSS": "Clipped",  # or "KL-DIVERGENCE"
    "CLIPPING_LOSS_RATIO": 0.1,
    "ENTROPY_LOSS_RATIO": 0.2,
    "CRITIC_LOSS": "mean_squared_error",
    "OPTIMIZER": "Adam",
    "TARGET_UPDATE_ALPHA": 0.9,
    }

    dic_env_conf = {
        "ENV_NAME": "LunarLander-v2",
        "GYM_SEED": 1,
        "LIST_STATE_NAME": ["state"],
        "ACTION_RANGE": "-1-1", # or "-1~1"
        "POSITIVE_REWARD": True
    }

    dic_path ={
        "PATH_TO_MODEL_actor": "/home/radia/catkin_ws/rl-mapless-navigation/src/trained_models/ppo/actor",
        "PATH_TO_MODEL_critic": "/home/radia/catkin_ws/rl-mapless-navigation/src/trained_models/ppo/critic"
    }

    dic_exp_conf = {
        "TRAIN_ITERATIONS": 100,
        "MAX_EPISODE_LENGTH": 1000,
        "TEST_ITERATIONS": 10
    }
    main()


