import random
from typing import Dict

from attribute import Attribute


class Node:
    def __init__(self, index, times) -> None:
        self.index = index
        self.value = None
        self.weights = None
        self.attribute = None
        self.times = times
        self.adjacents = []
        self.reset()

    def update_value(self, has_value=True):
        """Update value of node.

        Raises:
            NotImplementedError: Must implement this method.
        """
        raise NotImplementedError("You should implement this method before use it.")

    def update_weight(self, weights):
        """Update weights of node.

        Raises:
            NotImplementedError: Must implement this method.
        """
        raise NotImplementedError("You should implement this method before use it.")

    def _hard_distance(self):
        """Calc real distance between adjacent nodes.

        Returns:
            float: distance
        """
        if not self.weights:
            raise Exception("You should init weights before use it.")
        d = 0
        for adj, _ in self.weights:
            d += abs(adj.value - self.value)
        return d

    def _soft_distance(self):
        """Calc soft distance aka. weighted distance, between adjacent nodes.

        Returns:
            float: distance
        """
        if not self.weights:
            raise Exception("You should init weights before use it.")
        d = 0
        for adj, w in self.weights.items():
            d += abs(adj.value - self.value) * w
        return d

    def set_weight(self, weights: Dict):
        self.weights = weights

    def reset(self):
        # reset value to random value
        self.value = random.random() * self.times
        # reset weights to mean weight
        if self.weights:
            mean_w = 1 / len(self.weights)
            for adj, _ in self.weights.items():
                self.weights[adj] = mean_w

    def __str__(self) -> str:
        str = "Node {0}\tValue: {1}\n\n".format(self.index, self.value)
        for adj, w in self.weights.items():
            str += "Adj {0}\tWeight: {1}\n".format(adj.index, w)
        str += "-----------------------------------------------"
        return str


class NormalNode(Node):
    def __init__(self, index, times) -> None:
        super().__init__(index, times)
        self.attribute = Attribute.NORMAL

    def update_value(self, has_noise=True):
        """Update value of node.

        Args:
            has_noise (bool, optional): if or not has noise. Defaults to True.
        """
        m = 0
        for adj, w in self.weights.items():
            noise = (
                0 if not has_noise else (random.random() - 0.5) * 2 / 100 * self.times
            )
            m += w * (adj.value - self.value) + noise
        self.value += m

    def update_weight(self, weights):
        for i, adj in enumerate(self.weights.keys()):
            if adj.index == self.index:
                continue
            self.weights[adj] = weights[self.adjacents.index(adj.index)]
        # normalize weights
        self.normalize(sum(weights))

    def normalize(self, w_sum):
        if not self.weights:
            raise Exception("You should init weights first.")
        rest_total = 1 - (1 / len(self.weights))
        for adj, w in self.weights.items():
            if adj.index == self.index:
                continue
            self.weights[adj] = w / w_sum * rest_total


class RandomNode(Node):
    def __init__(self, index, times) -> None:
        super().__init__(index, times)
        self.attribute = Attribute.RANDOM

    def update_value(self, has_noise=True):
        self.value = random.random() * self.times

    def update_weight(self, weights):
        pass


class ConstantNode(Node):
    def __init__(self, index, times) -> None:
        """Constant node.

        Args:
            v (float): value
            times (int): times of A
        """
        super().__init__(index, times)
        self.attribute = Attribute.CONSTANT

    def update_value(self, has_value=True):
        pass

    def update_weight(self, weights):
        pass