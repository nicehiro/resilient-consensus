import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
    

device = torch.device('cpu')

# summary writer for tensorboard
writer = SummaryWriter('logs')


def normalize(data):
    """Normalize data.
    """
    data = np.array(data, dtype=np.float32)
    mean = data.mean()
    std = np.std(data)
    return (data - mean) / std


def batch_train(train, n=100):
    """Train a batch of episode, when they convergence or
    train steps get a certain number stop train.

    @:param train
        a method for training
    """
    failed_n = 0
    success_n = 0
    for i in range(n):
        t = train().matrix
        success = True
        for j in range(len(t)):
            if not success:
                break
            for k in range(0, 3):
                if t[j][k] > 0.1:
                    success = False
                    break
        if success:
            success_n += 1
        else:
            failed_n += 1
        print('Test Numbers: {0}\nSuccess Numbers: {1}\tFailed Numbers: {2}'.format(n, success_n, failed_n))
