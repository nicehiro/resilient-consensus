import random
from typing import Dict
from attribute import Attribute


class Node:
    def __init__(self, index, times, probs=1, seed=1, noise_scale=0.01) -> None:
        self.index = index
        self.value = None
        self.weights = None
        self.attribute = None
        self.times = times
        self.seed = seed
        self.noise_scale = noise_scale
        self.adjacents = []
        self.reset()

    def update_value(self, value=None):
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

    def update_weight_by_adj(self, adj, weights):
        if not adj:
            raise Exception("Adj should be a Node instance.")
        if not self.weights:
            raise Exception("You should init weights before use it.")
        if adj not in self.weights.keys():
            raise Exception("Adj is not node's adjacent.")
        self.weights[adj] = weights

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
        random.seed(self.seed)
        # reset value to random value
        self.value = random.random() * self.times
        # reset weights to mean weight
        if self.weights:
            mean_w = 1 / len(self.weights)
            for adj, _ in self.weights.items():
                self.weights[adj] = mean_w

    def normalize(self, w_sum):
        pass

    def __str__(self) -> str:
        str = "Node {0}\tValue: {1}\n\n".format(self.index, self.value)
        for adj, w in self.weights.items():
            str += "Adj {0}\tWeight: {1}\n".format(adj.index, w)
        str += "-----------------------------------------------"
        return str

    def update_value_normaly(self, value=None):
        """Update value of node.

        Args:
            has_noise (bool, optional): if or not has noise. Defaults to True.
        """
        m = 0
        for adj, w in self.weights.items():
            noise = self.noise_scale * (random.random() * 2 - 1) * self.times
            # noise = 0 if not has_noise else (random.random() * 2 - 1) / 100 * self.times
            w = 0 if w < 0.05 else w
            m += w * (adj.value - self.value) + noise
        self.value += m


class NormalNode(Node):
    def __init__(self, index, times, probs=1, seed=0, noise_scale=0.01) -> None:
        super().__init__(index, times, probs, seed, noise_scale)
        self.attribute = Attribute.NORMAL

    def update_value(self, value=None):
        """Update value of node.

        Args:
            has_noise (bool, optional): if or not has noise. Defaults to True.
        """
        self.update_value_normaly(value)

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
        for adj, w in self.weights.items():
            self.weights[adj] = w / w_sum


class RandomNode(Node):
    def __init__(self, index, times, probs=1, seed=1, noise_scale=0.01) -> None:
        super().__init__(index, times, probs, seed, noise_scale)
        self.attribute = Attribute.RANDOM
        self.probs = probs

    def update_value(self, value=None):
        if random.random() <= self.probs:
            self.value = random.random() * self.times
        else:
            self.update_value_normaly(value)

    def update_weight(self, weights):
        pass


class ConstantNode(Node):
    def __init__(self, index, times, probs=1, seed=2, noise_scale=0.01) -> None:
        """Constant node.

        Args:
            v (float): value
            times (int): times of A
        """
        super().__init__(index, times, probs, seed, noise_scale)
        self.attribute = Attribute.CONSTANT
        self.probs = probs
        random.seed(self.seed)
        self.constant_value = random.random() * self.times

    def update_value(self, value=None):
        if random.random() < self.probs:
            self.value = self.constant_value
        else:
            self.update_value_normaly(value)

    def update_weight(self, weights):
        pass


class IntelligentNode(Node):
    def __init__(self, index, times, probs=1, seed=3, noise_scale=0.01) -> None:
        """Intelligent node.

        Args:
            index (int): index
            times (int): times
        """
        super().__init__(index, times, probs, seed, noise_scale)
        self.attribute = Attribute.INTELLIGENT

    def update_value(self, value):
        self.value = value

    def update_weight(self, weights):
        pass