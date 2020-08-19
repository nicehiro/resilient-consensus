import numpy as np
import torch

from maddpg import core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, X_dim, A_dim):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.act2_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.X_buf = np.zeros(core.combined_shape(size, X_dim), dtype=np.float32)
        self.X2_buf = np.zeros(core.combined_shape(size, X_dim), dtype=np.float32)
        self.A_buf = np.zeros(core.combined_shape(size, A_dim), dtype=np.float32)
        self.A2_buf = np.zeros(core.combined_shape(size, A_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, next_act, done, X, X_, A, A_):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.act2_buf[self.ptr] = next_act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.X_buf[self.ptr] = X
        self.A_buf[self.ptr] = A
        self.X2_buf[self.ptr] = X_
        self.A2_buf[self.ptr] = A_
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     act2=self.act2_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     X=self.X_buf[idxs],
                     A=self.A_buf[idxs],
                     X_=self.X2_buf[idxs],
                     A_=self.A2_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
