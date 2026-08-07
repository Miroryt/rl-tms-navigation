"""Evaluate an already existing agent"""

import numpy as np
import argparse
import datetime
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
import itertools
import torch
from sac import SAC
from kuka_ros2_env import KukaRos2Env

"""Parse command line arguments"""
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="KUKA_ROS2_ENV",
                    help='Environment (default: KUKA_ROS2_ENV')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy every 10 episode (default: False)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.00005, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.65, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=50000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=0, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=0, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

"""Set the environment parameters. If demo=True, 'training' and 'goal' do not matter."""
# training -> used for NN training, no ROS2 node needed
# demo -> demo mode where user can sample random goal points from head mesh or define any custom coordinates
# goal -> options: 'moving_random', 'moving_head'
# collision -> Do we take end-effector collision to the head into account
# plots -> Do we create plots from training
# run_itself -> Runs num_eval_episodes amount of episodes and computes averages of certain data and plots them in the end
env = KukaRos2Env(training=False, goal='moving_head', collision=True, plots=True, run_itself=True, action_EE_coordinates=True) # initialize your environment

agent = SAC(env.observation_space.shape[0], env.action_space, args)  # initialize the agent object
agent.load_checkpoint("/home/user/rl-tms-navigation/src/checkpoints/sac_checkpoint_agent", evaluate=True) # Change the path to match your device
updates = 0


#How many episodes you want to see the agent perform:
num_eval_episodes = 50

for i in range(num_eval_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        action = agent.select_action(state, evaluate=True)  #ensure deterministic evaluation
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1

    print(f"Episode {i + 1} finished: total reward = {episode_reward}, steps = {steps}")


#Close the env after evaluation
env.close()
