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
                 memory_size=int(1e6),
                 batch_size=64,
                 gamma=0.99,
                 restore=False,
                 need_exploit=True,
                 train=True,
                 hidden_sizes=None,
                 save_model=False,
                 restore_path='./dqn/'):
        self.node_i = node_i
        self.features_n = features_n
        self.actions_n = actions_n
        self.gamma = gamma
        self.memory = Memory(self.features_n, self.actions_n // 2, memory_size)
        self.device = device
        self.dqn = DQN(self.features_n, self.actions_n,
                       hidden_sizes, activation=nn.ReLU)
        self.target_dqn = DQN(self.features_n, self.actions_n,
                              hidden_sizes, activation=nn.ReLU)
        self.dqn.to(device)
        self.target_dqn.to(device)
        self.target_dqn.eval()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.batch_size = batch_size
        self.restore_path = restore_path + 'dqn-{0}.pkl'.format(node_i)
        self.steps = 0
        self.eps_start = 0.9
        self.eps_end = 0.01
        self.eps_decay = 100000
        self.need_exploit = need_exploit
        self.train = train
        self.save_model = save_model
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
                state = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
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
        if not self.train or not self.memory.can_sample():
            return 0.0
        data = self.memory.sample(self.batch_size)
        obs, act, rew, obs_next = data['obs'], data['act'], data['rew'], data['obs_next']
        act_t = [[] for _ in range(len(act))]
        for i, act_ in enumerate(act):
            for j in range(len(act_)):
                act_t[i].append(j * 2 + act_[j].item())
        act_t = torch.tensor(act_t).to(device)
        q_eval = self.dqn(obs).to(device).gather(1, act_t)
        dim_a = act_t.shape[1]
        q_eval = q_eval.sum(1) / dim_a
        q_next_all = self.target_dqn(obs_next).to(device)
        q_next = [[] for _ in range(len(obs_next))]
        for i, q_n in enumerate(q_next_all):
            for j in range(0, len(q_n), 2):
                q_next[i].append(q_n[j].item() if q_n[j] > q_n[j+1] else q_n[j+1].item())
        q_next = torch.tensor(q_next).to(device)
        q_next = q_next.sum(1) / dim_a
        q_target = (q_next * self.gamma) + rew.squeeze()

        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 1000 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            print('Trained model saved successfully!')
            if self.save_model:
                self.save()
        return loss.mean()

    def save(self):
        torch.save(self.target_dqn.state_dict(), self.restore_path)

    def restore(self):
        params = torch.load(self.restore_path)
        self.target_dqn.load_state_dict(params)
        self.dqn.load_state_dict(params)
