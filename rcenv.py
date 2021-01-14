from attribute import Attribute
from topology import Topology
import gym
import numpy as np


class Env(gym.Env):
    def __init__(self, adj_matrix, node_attrs, times=1, has_noise=True) -> None:
        super().__init__()
        self.topology = Topology(adj_matrix, node_attrs, times)
        self.n = self.topology.n
        self.features_n = []
        self.actions_n = []
        self._calc_dim()

    def _calc_dim(self):
        """Calc features and actions dimention."""
        for node in self.topology.nodes:
            if node.attribute is not Attribute.NORMAL:
                self.features_n.append(None)
                self.actions_n.append(None)
            n = len(node.weights)
            self.features_n.append(n)
            self.actions_n.append(n - 1)

    def step(self, actions):
        """Execute actions for each node.

        Args:
            actions ([List]): List of actions for each node.
        """
        if len(actions) != self.n:
            raise Exception("actions size should be {0}".format(self.n))
        for node, action in zip(self.topology.nodes, actions):
            node.update_weight(action)
        for node in self.topology.nodes:
            node.update_value()

    def reset(self):
        states = []
        self.topology.reset()
        for i, node in enumerate(self.topology.nodes):
            state = np.zeros(shape=[self.features_n[i]])
            for adj, _ in node.weights:
                state.append(adj.value)
            states.append(state)
        return states

    def render(self):
        pass
