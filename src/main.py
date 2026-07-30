import argparse
import datetime
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import ReplayMemory
from gym.wrappers import TimeLimit

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
parser.add_argument('--lr', type=float, default=0.00003, metavar='G',
                    help='learning rate (default: 0.0003)') #0.00003
parser.add_argument('--alpha', type=float, default=0.65, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=0, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

"""Environment"""
from kuka_ros2_env import KukaRos2Env

"""----------------------------------"""
"""Set the environment parameters. If demo=True, 'training' and 'goal' do not matter."""
# training -> used for NN training, no ROS2 node needed
# goal -> options: 'moving_random', 'moving_head'
# collision -> Do we take end-effector collision to the head into account
# plots -> Do we create plots from training
#Collision=True slows training down
env = KukaRos2Env(training=True, goal='moving_head', collision=True, action_EE_coordinates=True)
"""----------------------------------"""

"""Set seeds for reproducibility"""
#env.seed(args.seed)

state, _ = env.reset()       # seeds env’s RNG
#env.action_space.seed(args.seed)           # seeds action sampling
#torch.manual_seed(args.seed)
#np.random.seed(args.seed)

"""Initialize Agent with state-dim, action-space and hyperparameters"""
"""Do this even if you want to load an agent"""
agent = SAC(env.observation_space.shape[0], env.action_space, args)

"""To load an already existing agent:"""
#agent.load_checkpoint(ckpt_path="~/rl-tms-navigation/src/checkpoints/sac_checkpoint_agent", evaluate=False)

"""Initialize TensorBoard writer to log losses and rewards over time"""
writer = SummaryWriter('runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                              "autotune" if args.automatic_entropy_tuning else ""))

"""Initialize Replay Buffer to store and sample transitions"""
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

reward_history = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    """Reset the environment (observe state)"""
    state, _ = env.reset()

    """The first args.start_steps is spent sampling random actions, after that, we sample from policy"""
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                #Update summary writer:
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                writer.add_scalar('Joint violations', env.num_violations, updates)
                writer.add_scalar('Norm of distance difference', env.pos_err_field, updates)
                writer.add_scalar('Norm of orientation difference', env.angle_err_field, updates)
                if env.collision==True:
                   writer.add_scalar('Collided: ', env.collision_field, updates)
                updates += 1

        #gymnasium has 5 parameters involved in .step()
        next_state, reward, terminated, truncated, _, = env.step(action)
        done = terminated or truncated

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        reward_history.append(episode_reward)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_episode_steps else float(not done)
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("----------------------------------------")
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    print("----------------------------------------")


"""Save model"""
agent.save_checkpoint(env_name="SAC_KUKA_ROS2", suffix="gen13_5_adaptive")

env.close()

"""To view figures: 
open terminal and type:
'tensorboard --logdir=runs'
open browser and type:
http://localhost:6006
"""

