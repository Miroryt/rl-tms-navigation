import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        """
        :param num_inputs: state/input dimension / observation space
        :param action_space: action space of the environment
        :param args: contains all hyperparameters (gamma, alpha, lr etc.)
        """

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        """Set device to GPU or otherwise CPU:"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("\n Device set to: ", self.device, "\n")

        """Initialize Critic Network with Adam optimizer"""
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        """Initialize target Critic Network"""
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, evaluate=False):
        """action selection; during training samples actions from the stochastic policy
                    during evaluation uses the mean action"""
        """Converts NumPy state to a PyTorch tensor and back again for processing"""
        #state = torch.FloatTensor(np.array(state)).to(self.device).unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state) #Uses stochastic policy
        else:
            _, _, action = self.policy.sample(state) #Uses mean
        return action.detach().cpu().numpy()[0]


    def update_parameters(self, memory, batch_size, updates):
        """This is the core learning method
        Updates NN parameters with backpropagation
        """

        """Sample a batch from memory"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        """Convert to tensors"""
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        """Compute target Q-values using the target critic network"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        """Compute current Q estimates for the taken actions"""
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        #Debug prints to show vector dimensions:

        #print('reward_batch shape:', reward_batch.shape)
        #print('mask_batch shape:', mask_batch.shape)
        #print('min_qf_next_target shape:', min_qf_next_target.shape)
        #print('next_state_log_pi shape:', next_state_log_pi.shape)
        #print('next_state_action shape:', next_state_action.shape)

        """compute critic loss between predicted Q and target Q with element-wise mean-squared error (MSE)"""
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~M[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~M[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        """backpropagate"""
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        """Compute policy loss"""
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼M,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        """backpropagate policy loss"""
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        """Automatic entropy (alpha parameter) tuning if enabled in args"""
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        """Return losses for logging"""

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        """Save model parameters"""
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        """Load model parameters"""
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

