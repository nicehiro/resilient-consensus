from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import time
from ddpg.core import MLPActorCritic
from ddpg.memory import ReplayBuffer
from gym.spaces import Box, Discrete
from utils import normalize


class Agent:
    def __init__(self,
                 node_i,
                 observation_space,
                 action_space,
                 actor_lr=1e-5,
                 critic_lr=1e-4,
                 memory_size=100,
                 gamma=0.99,
                 polyak=0.95,
                 noise_scale=0.1,
                 batch_size=4):
        self.node_i = node_i
        self.obs_dim = observation_space.shape[0]
        if isinstance(observation_space, Box):
            self.act_dim = action_space.shape[0]
        elif isinstance(observation_space, Discrete):
            self.act_dim = action_space.n
        self.action_space = action_space
        self.ac = MLPActorCritic(observation_space, action_space, [512, 512], nn.ReLU)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.memory = ReplayBuffer(self.obs_dim, self.act_dim, memory_size)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=actor_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.noise_scale = noise_scale
        self.act_limit = action_space.high[0]
        self.batch_size = batch_size

    def act(self, obs):
        a = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
        # a += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, 0, self.act_limit)

    def optimize(self):
        # Set up function for computing DDPG Q-loss
        def compute_loss_q(data):
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

            q = self.ac.q(o, a)
            # Bellman backup for Q function
            with torch.no_grad():
                q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
                backup = r + self.gamma * (1 - d) * q_pi_targ

            # MSE loss against Bellman backup
            loss_q = ((q - backup) ** 2).mean()

            # Useful info for logging
            loss_info = dict(QVals=q.detach().numpy())

            return loss_q, loss_info

        # Set up function for computing DDPG pi loss
        def compute_loss_pi(data):
            o = data['obs']
            q_pi = self.ac.q(o, self.ac.pi(o))
            return -q_pi.mean()

        def update(data):
            # First run one gradient descent step for Q.
            self.q_optimizer.zero_grad()
            loss_q, loss_info = compute_loss_q(data)
            loss_q.backward()
            self.q_optimizer.step()

            # Freeze Q-network so you don't waste computational effort
            # computing gradients for it during the policy learning step.
            for p in self.ac.q.parameters():
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-network so you can optimize it at next DDPG step.
            for p in self.ac.q.parameters():
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
            return loss_q, loss_pi
        batch = self.memory.sample_batch(self.batch_size)
        return update(batch)
