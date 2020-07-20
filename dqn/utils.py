import numpy as np
import torch

from utils import device


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class Memory:
    def __init__(self, obs_dim, act_dim, size):
        self.size = size
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.int)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.obs_next_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.ptr = 0
        self.p = 0

    def __len__(self):
        return min(self.size, self.p)

    def store(self, obs, act, rew, obs_next):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs_next_buf[self.ptr] = obs_next
        self.ptr = (self.ptr + 1) % self.size
        self.p += 1

    def sample(self, sample_size=64):
        idx = np.random.randint(0, min(self.size, self.p), sample_size)
        data = dict(obs=self.obs_buf[idx],
                    act=self.act_buf[idx],
                    rew=self.rew_buf[idx],
                    obs_next=self.obs_next_buf[idx])
        return {k: torch.as_tensor(v, dtype=torch.long if k == 'act' else torch.float32).to(device) for k, v in data.items()}

    def can_sample(self):
        return self.p >= int(5e4)
