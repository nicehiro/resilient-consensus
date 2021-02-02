import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from analysis.save import save_nodes_value

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

adjacent_matrix = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
]


adjacent_matrix = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
]


def summary_writer(log_path="log"):
    # summary writer for tensorboard
    return SummaryWriter("logs")


def normalize(data):
    """Normalize data."""
    data = np.array(data, dtype=np.float32)
    mean = data.mean()
    std = np.std(data)
    return (data - mean) / std


def batch_train(train, method="DQN", label="1r2c", n=1000):
    """Train a batch of episode, when they convergence or
    train steps get a certain number stop train.

    @:param train
        a method for training
    @:param method
        train method description
    @:param label
        situation
    @:param n
        train times
    """
    failed_n = 0
    success_n = 0
    for i in range(n):
        env = train()
        if env.is_done():
            success_n += 1
        else:
            failed_n += 1
        print(env.map.node_val())
        print(
            "Test Numbers: {0}\nSuccess Numbers: {1}\tFailed Numbers: {2}".format(
                n, success_n, failed_n
            )
        )
        save_nodes_value(
            env.map, method, env.is_done(), label, path="data/nodes_values.csv"
        )
