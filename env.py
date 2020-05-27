import math
import random
from enum import Enum
from typing import List

import numpy as np


class Property(Enum):
    """Property of Node.
    """
    GOOD = 1
    # with random value every time
    RANDOM = 2
    # with constant value every time
    CONSTANT = 3


class Node:
    def __init__(self, index, v, p: Property):
        self.index = index
        # node value
        self.v = v
        # weight  index: weight
        self.weights = dict()
        # property of node
        self.property = p

    def __calc_neighbors(self):
        return len(self.weights)

    @property
    def neighbors_n(self):
        return len(self.weights)


class Map:
    def __init__(self, nodes: List[Node]):
        """Map contains a lot of Nodes.
        """
        self.nodes = nodes
        self.nodes_n = len(nodes)
        self.matrix = self.__adja_matrix()
        self.weights_index = [[] for _ in range(self.nodes_n)]
        self.__calc_weights_index()

    def __adja_matrix(self):
        """Adjacency matrix.
        """
        m = np.zeros(shape=(self.nodes_n, self.nodes_n),
                     dtype=np.float32)
        for i, node in enumerate(self.nodes):
            for v, w in node.weights.items():
                m[i][v] = w
        return m

    def update_by_times(self, node_i, neigh_i, w_multi):
        """Update node_i's neighbor neigh_i's weight.
        """
        self.nodes[node_i].weights[neigh_i] *= w_multi
        self.matrix[node_i][neigh_i] *= w_multi

    def update_by_weight_index_and_times(self, node_i, weights_i, w_multi):
        """Update node_i's neigh_i's weight.

        But we only have weights index.
        """
        self.update(node_i,
                    self.weights_index[node_i][weights_i],
                    w_multi)

    def update_by_weight(self, node_i, neigh_i, weight):
        self.nodes[node_i].weights[neigh_i] = weight
        self.matrix[node_i][neigh_i] = weight

    def update_by_weight_index(self, node_i, weight_i, weight):
        self.update_by_weight(node_i,
                              self.weights_index[node_i][weight_i],
                              weight)

    def normalize(self, node_i):
        """Normalize node weight make them sum to 1.
        """
        for i in range(self.nodes_n):
            s = self.matrix[node_i].sum()
            self.matrix[node_i][i] /= s
            if self.nodes[node_i].weights.get(i):
                self.nodes[node_i].weights[i] /= s

    def update_value_of_node(self):
        """Update value of every node.
        """
        for node in self.nodes:
            if node.property == Property.GOOD:
                m = 0
                for i, w in node.weights.items():
                    m += w * (self.nodes[i].v - node.v)
                node.v += m
            elif node.property == Property.CONSTANT:
                pass
            elif node.property == Property.RANDOM:
                node.v = random.random()

    def states(self):
        """Get state of each node.

        State contains node's value & node's weights & node's adja values.
        """
        states = [[] for _ in range(self.nodes_n)]
        for i, node in enumerate(self.nodes):
            states[i].append(node.v)
            for v, w in node.weights.items():
                states[i].append(w)
                states[i].append(self.nodes[v].v)
        return states

    def __calc_weights_index(self):
        for i in range(self.nodes_n):
            for j in range(self.nodes_n):
                if self.matrix[i][j] > 0:
                    self.weights_index[i].append(j)

    def __len__(self):
        return self.nodes.__len__()

    def __str__(self):
        str = ''
        for i, node in enumerate(self.nodes):
            str += 'Node: {0}\tValue: {1}\n'.format(i, node.v)
            for v, w in node.weights.items():
                str += 'Adj: {0}\tWeight: {1}\n'.format(v, w)
            str += '\n'
        return str


class Env:

    def __init__(self, nodes_n):
        self.nodes_n = nodes_n
        self.map, self.features_n, self.outputs_n = self.make_map()

    def reset(self):
#         self.map, self.features_n, self.outputs_n = self.make_map()
        return self.states()

    def make_map(self):
        node_random = Node(0, random.random(), Property.RANDOM)
        node_constant_1 = Node(1, random.random(), Property.CONSTANT)
        node_constant_2 = Node(2, random.random(), Property.CONSTANT)
        nodes = [node_random, node_constant_1, node_constant_2] + \
                [Node(i, random.random(), Property.GOOD) for i in range(3, self.nodes_n)]
        nodes[0].weights = {0: 1}
        nodes[1].weights = {1: 1}
        nodes[2].weights = {2: 1}
        nodes[3].weights = {1: 0.25, 3: 0.25, 5: 0.25, 9: 0.25}
        nodes[4].weights = {2: 0.25, 4: 0.25, 6: 0.25, 8: 0.25}
        nodes[5].weights = {2: 0.33, 5: 0.33, 7: 0.33}
        nodes[6].weights = {0: 0.25, 3: 0.25, 4: 0.25, 6: 0.25}
        nodes[7].weights = {0: 0.2, 1: 0.2, 4: 0.2, 6: 0.2, 7: 0.2}
        nodes[8].weights = {1: 0.25, 5: 0.25, 7: 0.25, 8: 0.25}
        nodes[9].weights = {2: 0.25, 3: 0.25, 7: 0.25, 9: 0.25}
        features_n = [x.neighbors_n * 2 + 1 for x in nodes]
        outputs_n = [x.neighbors_n * 2 for x in nodes]
        return Map(nodes), features_n, outputs_n

    def step(self, action, node_i, is_continuous=False):
        """Update node_i's weights.
        """
        # update weight
        if is_continuous:
            action = action[0]
            for i in range(len(action)):
                self.map.update_by_weight_index(node_i, i, action[i])
        else:
            for i in range(len(action)):
                if action[i] == 1:
                    # down
                    self.map.update_by_weight_index_and_times(node_i, i, 0.98)
                else:
                    # up
                    self.map.update_by_weight_index_and_times(node_i, i, 1.02)
        self.map.normalize(node_i)
        # rewards
        r = 0
        for i in range(self.nodes_n):
            r += self.map.matrix[node_i][i] * abs(self.map.nodes[node_i].v - self.map.nodes[i].v)
        r = math.exp(-20 * r)
        r = (r - 0.5) / 100
        return r

    def update_value_of_node(self):
        """When synchronize update node, use this function when
         all node's updating finished.
         When asynchronous update node, use this function when
         each node's updating finished.
        """
        self.map.update_value_of_node()

    def states(self):
        """Get current states.
        """
        return self.map.states()
