import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn.model import DQN
from dqn.utils import Memory


class DQNAgent:
    def __init__(self,
                 node_i,
                 features_n,
                 actions_n,
                 lr=0.001,
                 memory_size=500,
                 batch_size=64,
                 gamma=0.99,
                 restore=False,
                 restore_path='./dqn.pkl'):
        self.node_i = node_i
        self.features_n = features_n
        self.actions_n = actions_n
        self.gamma = gamma
        self.memory = Memory(self.features_n, self.actions_n//2, memory_size)
        self.device = 'cpu'
        self.dqn = DQN(self.features_n, self.actions_n,
                       [64, 64], activation=nn.ReLU)
        self.target_dqn = DQN(self.features_n, self.actions_n,
                              [64, 64], activation=nn.ReLU)
        self.target_dqn.eval()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.batch_size = batch_size
        self.restore_path = restore_path
        self.steps = 0
        if restore:
            self.restore()

    def act(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            outputs = self.dqn(obs)
            actions = []
            for i in range(0, len(outputs), 2):
                up = outputs[i].item()
                down = outputs[i + 1].item()
                actions.append(0 if up > down else 1)
            return actions, outputs

    def optimize_model(self):
        if not self.memory.can_sample():
            return 0.0
        self.steps += 1
        data = self.memory.sample(self.batch_size)
        obs, act, rew, obs_next = data['obs'], data['act'], data['rew'], data['obs_next']
        q_eval = self.dqn(obs).gather(1, act)
        q_next = self.target_dqn(obs_next).max(1)[0].detach()
        q_target = (q_next * self.gamma) + rew.squeeze()

        loss = F.mse_loss(q_eval, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 100 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.save()
        return loss.mean()

    def save(self):
        torch.save(self.target_dqn.state_dict(), self.restore_path)

    def restore(self):
        params = torch.load(self.restore_path)
        self.target_dqn.load_state_dict(params)
        self.dqn.load_state_dict(params)
