import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

"""bounds for the policy's log-standard deviation"""
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6 #A small constant to prevent taking log(0)

def weights_init_(m):
    """Applies Xavier uniform initialization to all nn.Linear layers' weights and zeroes their biases"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    """The value network estimates the value V(s) of a state s under the current policy. Takes only state vector as input.
    Value network is not needed, and is not used in this implementation"""
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        """Input layer with input dimension num_inputs"""
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        """Three hidden layers of size hidden_dim"""
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        """Final linear layer to a single scalar output V(s)"""
        self.linear5 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        """Applies three ReLU-activated layers and outputs a scalar value"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Critic Network of two Q-Networks. The Critic network takes both state and action vectors as input and outputs two Q-values
        :param num_inputs: the dimensions of the state vector
        :param num_actions: the dimensions of the action vector
        :param hidden_dim: hidden dim
        """
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        """Computes two separate Q-values and return them as a tuple (later we take the min. of the two)"""
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)

        x2 = F.relu(self.linear5(xu))
        x2 = F.relu(self.linear6(x2))
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)

        return x1, x2

class GaussianPolicy(nn.Module):
    """Action Network (Policy Network). Takes state vector as input and outputs Gaussian distribution over actions"""
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        """The Action network outputs actions using Gaussian distribution and then applies tanh to squeeze the values into (-1, 1).
        Some environments however use action spaces with different bounds (-2, 2) etc., so to ensure the actions are valid for the
        environment, rescaling is done"""
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """Forward-pass outputs mu and log_std of the Gaussian distribution"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        """Stochastic action to execute:"""
        """This is used when training"""
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        """Mean is used when evaluating a network, that's why it behaves differently"""
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)