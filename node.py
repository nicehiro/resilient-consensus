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
        self.reset()

    def update_value(self):
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

    def set_weight(self, weights: Dict):
        self.weights = weights

    def reset(self):
        # reset value to random value
        self.value = random.random() * self.times
        # reset weights to mean weight
        if self.weights:
            mean_w = 1 / len(self.weights)
            for adj, _ in self.weights:
                self.weights[adj] = mean_w


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
        for adj, w in self.weights:
            noise = (
                0 if not has_noise else (random.random() - 0.5) * 2 / 100 * self.times
            )
            m += w * (adj.value - self.value) + noise
        self.value += m

    def update_weight(self, weights):
        i = 0
        for adj, _ in self.weights:
            if adj.index == self.index:
                continue
            self.weights[adj] = weights[i]
            i += 1
        self.normalize()

    def normalize(self):
        rest_total = 1 - (1 / n)
        for adj, w in self.weights:
            if adj.index == self.index:
                continue


class RandomNode(Node):
    def __init__(self, index, times) -> None:
        super().__init__(index, times)
        self.attribute = Attribute.RANDOM

    def update_value(self):
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

    def update_value(self):
        pass

    def update_weight(self, weights):
        pass