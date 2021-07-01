from modified_rl.core import MLPActor
from modified_rl.memory import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        buffer_size,
        lr,
        gamma,
        node_i,
        restore_path,
        batch_size,
        value,
        node_index_of_weights,
    ) -> None:
        super().__init__()
        self.memory = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=buffer_size)
        self.policy = MLPActor(
            obs_dim=obs_dim + act_dim,
            act_dim=act_dim,
            hidden_sizes=[256],
            activation=nn.ReLU,
            act_limit=1,
        )
        self.node_i = node_i
        self.value = value
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.node_index_of_weights = node_index_of_weights
        self.restore_path = restore_path + "{0}".format(self.node_i)

    def update_value(self, value):
        self.value = value

    def act(self, obs, act):
        """
        Get action(weights).
        """
        inputs = torch.cat(
            (
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(act, dtype=torch.float32),
            )
        )
        a = self.policy(inputs)
        a = F.softmax(a, dim=0)
        return a

    def optimize(self):
        """
        Optimize network.
        """
        data = self.memory.sample_batch(self.batch_size)
        o, a, r, o_, _ = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        self.optimizer.zero_grad()
        # v = o[:, self.node_index_of_weights].unsqueeze(dim=-1)
        # d = torch.sum(abs(o[:, self.node_index_of_weights + 1 :] - v) * a, dim=1)
        d = torch.sum(abs(o) * a, dim=1)
        r = torch.exp(-20 * d)
        loss = -torch.mean(r)
        loss.backward()
        for params in self.policy.parameters():
            params.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def save(self):
        torch.save(self.policy.state_dict(), self.restore_path)

    def restore(self):
        params = torch.load(self.restore_path)
        self.policy.load_state_dict(params)