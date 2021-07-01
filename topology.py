from typing import List

from attribute import Attribute
from node import ConstantNode, IntelligentNode, Node, NormalNode, RandomNode


class Topology:
    """Describe relationships between nodes and edges."""

    def __init__(
        self,
        adjacency_matrix: List[List],
        node_attrs: List,
        probs: List,
        seeds: List,
        times=1,
        noise_scale=0,
    ) -> None:
        """Topology initialization.

        Args:
            adjacency_matrix (List[List]): adjacency matrix of topology.
            node_attrs (Dict): node attribute.
            times: default times is 1, means node value between 0 to 1.
            seeds: list of n random seeds
        """
        self.nodes = list()
        self.times = times
        self.n = len(adjacency_matrix)
        # self.seeds = [i for i in range(self.n)]
        self.seeds = seeds
        self.noise_scale = noise_scale
        self._generate_topo(adjacency_matrix, node_attrs, probs)

    def _generate_topo(self, adj_maxtrix: List[List], node_attrs: List, probs: List):
        """Generate topology.

        Args:
            adj_maxtrix (List[List]): adjacency matrix of topology.
            node_attrs (Dict): node attribute.
        """
        if len(adj_maxtrix) != len(node_attrs):
            raise Exception(
                "The length of matrix should be as same as the length of node_types."
            )
        for i, attr in enumerate(node_attrs):
            if attr is Attribute.NORMAL:
                node = NormalNode(
                    i,
                    times=self.times,
                    seed=self.seeds[i],
                    noise_scale=self.noise_scale,
                )
            elif attr is Attribute.RANDOM:
                node = RandomNode(
                    i,
                    times=self.times,
                    probs=probs[i],
                    seed=self.seeds[i],
                    noise_scale=self.noise_scale,
                )
            elif attr is Attribute.CONSTANT:
                node = ConstantNode(
                    i,
                    times=self.times,
                    probs=probs[i],
                    seed=self.seeds[i],
                    noise_scale=self.noise_scale,
                )
            elif attr is Attribute.INTELLIGENT:
                node = IntelligentNode(
                    i,
                    times=self.times,
                    seed=self.seeds[i],
                    noise_scale=self.noise_scale,
                )
            self.nodes.append(node)
        for i, adjs in enumerate(adj_maxtrix):
            node = self.nodes[i]
            n = sum(adjs)
            weights = {}
            adjacents = []
            for j, adj in enumerate(adjs):
                if adj == 1:
                    weights[self.nodes[j]] = 1 / n
                    if j != i:
                        adjacents.append(j)
            node.adjacents = adjacents
            node.set_weight(weights)

    def reset(self):
        for node in self.nodes:
            node.reset()

    def __str__(self) -> str:
        str = ""
        for node in self.nodes:
            str += node.__str__() + "\n"
        return str

    def update_value(self):
        for node in self.nodes:
            node.update_value()

    def hard_distance(self):
        """Calc real distance between each node's value.

        Returns:
            float: distance
        """
        dis = 0
        times = 0
        for i in range(self.n):
            for j in range(i, self.n):
                if (
                    self.nodes[i].attribute == Attribute.NORMAL
                    and self.nodes[j].attribute == Attribute.NORMAL
                ):
                    dis += abs(self.nodes[i].value - self.nodes[j].value)
                    times += 1
        return dis / times

    def node_val(self):
        res = {}
        for i, node in enumerate(self.nodes):
            res["Node{0}".format(i)] = node.value
        return res
