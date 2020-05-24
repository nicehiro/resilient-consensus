import torch.nn as nn


def build_net(sizes, activation, output_activation=nn.Identity):
    """Build fully connected layer neural network.
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act())
    return nn.Sequential(*layers)


class DQN(nn.Module):
    def __init__(self, features_n, outputs_n, hidden_sizes, activation):
        super(DQN, self).__init__()
        sizes = [features_n] + hidden_sizes + [outputs_n]
        self.dqn = build_net(sizes, activation, output_activation=nn.Softmax)

    def forward(self, obs):
        return self.dqn(obs).squeeze()
