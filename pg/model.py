import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Discrete, Box


def build_net(sizes, activation, output_activation=nn.Identity):
    """Build fully connected layer neural network.
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_probs(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_probs(pi, act)
        return pi, logp_a


class CategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(CategoricalActor, self).__init__()
        self.logits = build_net([obs_dim] + hidden_sizes + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits(obs)
        return Categorical(logits=logits)

    def _log_probs(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(GaussianActor, self).__init__()
        self.mu = build_net([obs_dim] + hidden_sizes + [act_dim], activation, nn.Tanh)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs):
        mu = self.mu(obs)
#         mu = 1 / 2 * (mu + 1)
        log_std = torch.exp(self.log_std)
        return Normal(mu, log_std)

    def _log_probs(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(Critic, self).__init__()
        self.v = build_net([obs_dim] + hidden_sizes + [1], activation)

    def forward(self, obs):
        return self.v(obs).squeeze()


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, activation):
        super(ActorCritic, self).__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(observation_space, Box):
            act_dim = action_space.shape[0]
            self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        elif isinstance(observation_space, Discrete):
            act_dim = action_space.n
            self.actor = CategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
        self.critic = Critic(obs_dim, hidden_sizes, activation)
        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def step(self, obs):
        with torch.no_grad():
            pi = self.actor._distribution(obs)
            a = pi.sample()
            logp_a = self.actor._log_probs(pi, a)
            v = self.critic(obs)
        return a.clamp(0.001, 1.0).numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


