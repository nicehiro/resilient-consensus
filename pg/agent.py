import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import time
from pg.model import ActorCritic
from pg.utils import VPGBuffer
from gym.spaces import Box, Discrete
from utils import writer


class Agent:
    def __init__(self,
                 node_i,
                 observation_space,
                 action_sapce,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 memory_size=1000,
                 train_v_iters=50,
                 gamma=0.99,
                 lam=0.95):
        self.node_i = node_i
        self.obs_dim = observation_space.shape[0]
        if isinstance(observation_space, Box):
            self.act_dim = action_sapce.shape[0]
        elif isinstance(observation_space, Discrete):
            self.act_dim = action_sapce.n
        self.ac = ActorCritic(observation_space, action_sapce, [64, 64, 64], nn.ReLU)
        self.memory = VPGBuffer(self.obs_dim, self.act_dim, memory_size, gamma=gamma, lam=lam)
        self.actor_optim = Adam(self.ac.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.ac.critic.parameters(), lr=critic_lr)
        self.train_v_iters = train_v_iters
#         writer.add_graph(self.ac, torch.zeros([1, self.obs_dim]))

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.ac.step(obs)

    def optimize(self):
        if not self.memory.can_optim():
            return 0, 0
        # Set up function for computing VPG policy loss
        def compute_loss_pi(data):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
            # Policy loss
            pi, logp = self.ac.actor(obs, act)
            loss_pi = -(logp * adv).mean()
            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            pi_info = dict(kl=approx_kl, ent=ent)
            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v(data):
            obs, ret = data['obs'], data['ret']
            return ((self.ac.critic(obs) - ret) ** 2).mean()

        data = self.memory.get()
        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        self.actor_optim.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        self.actor_optim.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.critic_optim.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            self.critic_optim.step()

        return loss_pi.item(), loss_v.item()
