import math
import random

import gym
import numpy as np

from attribute import Attribute
from topology import Topology


class Env(gym.Env):
    def __init__(self, adj_matrix, node_attrs, times=1, has_noise=True) -> None:
        super().__init__()
        self.topology = Topology(adj_matrix, node_attrs, times)
        self.n = self.topology.n
        self.features_n = []
        self.actions_n = []
        self.has_noise = has_noise
        self._calc_dim()

    def _calc_dim(self):
        """Calc features and actions dimention."""
        for node in self.topology.nodes:
            if node.attribute is not Attribute.NORMAL:
                self.features_n.append(None)
                self.actions_n.append(None)
                continue
            n = len(node.weights)
            self.features_n.append(n)
            self.actions_n.append(n - 1)

    def step(self, actions):
        """Execute actions for each node.

        Return reward, next_state.

        Args:
            actions ([List]): List of actions for each node.
        """
        if len(actions) != self.n:
            raise Exception("actions size should be {0}".format(self.n))
        for node, action in zip(self.topology.nodes, actions):
            node.update_weight(action)
        for node in self.topology.nodes:
            node.update_value()
        return self._reward(), self._state()

    def reset(self):
        """Reset env.

        Reset node value to random, weights to mean weights.
        """
        self.topology.reset()
        return self._state()

    def _state(self):
        states = []
        for i, node in enumerate(self.topology.nodes):
            if not self.features_n[i]:
                states.append(None)
                continue
            state = np.zeros(shape=[self.features_n[i]])
            state[0] = node.value
            for j, adj_index in enumerate(node.adjacents):
                state[j + 1] = self.topology.nodes[adj_index].value
            states.append(state)
        return states

    def render(self):
        """TODO: Draw image of current topology.

        Image contains node, node value, adj weights.
        """
        pass

    def _reward(self):
        """Calc rewards for all node."""
        rewards = []
        for node in self.topology.nodes:
            if node.attribute is Attribute.NORMAL:
                d = node._soft_distance()
                r = 1 * (math.exp(-20 * d) - 0.5)
                rewards.append(r)
            rewards.append(None)
        return rewards


if __name__ == "__main__":
    adj_matrix = [
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
    node_attrs = [
        Attribute.RANDOM,
        Attribute.CONSTANT,
        Attribute.RANDOM,
        Attribute.CONSTANT,
    ] + [Attribute.NORMAL for _ in range(8)]
    env = Env(adj_matrix, node_attrs)
    s = env.reset()
    while True:
        print(env.topology)
        actions = [
            [random.random() for _ in range(sum(adj_matrix[i]) - 1)] for i in range(12)
        ]
        s_, r = env.step(actions)
        s = s_
