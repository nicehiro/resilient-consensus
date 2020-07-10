import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn.model import DQN
from dqn.utils import Memory
from utils import device


class DQNAgent:
    def __init__(self,
                 node_i,
                 features_n,
                 actions_n,
                 lr=0.001,
                 memory_size=10,
                 batch_size=4,
                 gamma=0.99,
                 restore=False,
                 need_exploit=True,
                 restore_path='./dqn/dqn.pkl'):
        self.node_i = node_i
        self.features_n = features_n
        self.actions_n = actions_n
        self.gamma = gamma
        self.memory = Memory(self.features_n, self.actions_n // 2, memory_size)
        self.device = device
        self.dqn = DQN(self.features_n, self.actions_n,
                       [64, 64, 64, 64], activation=nn.ReLU)
        self.target_dqn = DQN(self.features_n, self.actions_n,
                              [64, 64, 64, 64], activation=nn.ReLU)
        self.target_dqn.eval()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.batch_size = batch_size
        self.restore_path = restore_path
        self.steps = 0
        self.eps_start = 0.9
        self.eps_end = 0.01
        self.eps_decay = 100000
        self.need_exploit = need_exploit

        if restore:
            self.restore()

    def act(self, obs):
        sample = random.random()
        # chose action randomly at the beginning, then slowly chose max Q_value
        eps_threhold = self.eps_end + (self.eps_start - self.eps_end) * \
                       math.exp(-1. * self.steps / self.eps_decay) \
            if self.need_exploit else 0.01
        self.steps += 1
        if sample > eps_threhold:
            with torch.no_grad():
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                outputs = self.dqn(state)
                actions = []
                for i in range(0, len(outputs), 2):
                    up = outputs[i].item()
                    down = outputs[i + 1].item()
                    actions.append(0 if up > down else 1)
                    # print(actions)
                return actions
        return [random.randrange(2) for _ in range(self.actions_n // 2)]

    def optimize_model(self):
        if not self.memory.can_sample():
            return 0.0
        data = self.memory.sample(self.batch_size)
        obs, act, rew, obs_next = data['obs'], data['act'], data['rew'], data['obs_next']
        act_t = [[] for _ in range(len(act))]
        for i, act_ in enumerate(act):
            for j in range(len(act_)):
                act_t[i].append(j * 2 + act_[j].item())
        act_t = torch.tensor(act_t)
        q_eval = self.dqn(obs).gather(1, act_t)
        dim_a = act_t.shape[1]
        q_eval = q_eval.sum(1) / dim_a
        q_next = self.target_dqn(obs_next).max(1)[0].detach()
        q_target = (q_next * self.gamma) + rew.squeeze()

        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 1000 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.save()
        return loss.mean()

    def save(self):
        torch.save(self.target_dqn.state_dict(), self.restore_path)

    def restore(self):
        params = torch.load(self.restore_path)
        self.target_dqn.load_state_dict(params)
        self.dqn.load_state_dict(params)
