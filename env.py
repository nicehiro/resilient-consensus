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
    # with special intelligence every time
    RIVAL = 4
    # with maddpg agent
    MADDPG = 5


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

    def update_v(self, action):
        """If action is 1: increase value a little
        If action is 0: decrease value a little.
        """
        if action == 1:
            self.v *= 1.1
        elif action == 0:
            self.v *= 0.9

    def update_v_directly(self, value):
        self.v = value


class Map:
    def __init__(self, nodes: List[Node], times=1):
        """Map contains a lot of Nodes.
        """
        self.nodes = nodes
        self.times = times
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
        self.update_by_times(node_i,
                             self.weights_index[node_i][weights_i],
                             w_multi)

    def update_by_weight(self, node_i, neigh_i, weight):
        self.nodes[node_i].weights[neigh_i] = weight
        self.matrix[node_i][neigh_i] = weight

    def update_by_weight_index(self, node_i, weight_i, weight):
        if node_i == self.weights_index[node_i][weight_i]:
            return
        self.update_by_weight(node_i,
                              self.weights_index[node_i][weight_i],
                              weight)

    def normalize(self, node_i):
        """Normalizing node weights to make them sum to 1.
        """
        s = self.matrix[node_i].sum() - self.matrix[node_i][node_i]
        l = len(self.nodes[node_i].weights)
        base_w = 1 / l
        others_w = 1 - base_w
        for i in range(self.nodes_n):
            if i == node_i:
                self.matrix[node_i][i] = base_w
                self.nodes[node_i].weights[i] = base_w
                continue
            self.matrix[node_i][i] = self.matrix[node_i][i] * others_w / s
            if self.nodes[node_i].weights.get(i):
                self.nodes[node_i].weights[i] = self.nodes[node_i].weights[i] * others_w / s
                
    def update_value_directly(self, node_i, value):
        """Update node_i value directly.
        """
        self.nodes[node_i].update_v_directly(value)

    def update_value_of_node(self, with_noise=False):
        """Update value of every node.
        """
        for node in self.nodes:
            if node.property == Property.GOOD:
                m = 0
                for i, w in node.weights.items():
                    m += w * (self.nodes[i].v - node.v)
                noise = (random.random() - 0.5) * 2 * self.times * 0.01 if with_noise else 0
                node.v += m + noise
            elif node.property == Property.CONSTANT:
                pass
            elif node.property == Property.RANDOM:
                node.v = random.random() * self.times
            elif node.property == Property.RIVAL:
                pass
            elif node.property == Property.MADDPG:
                pass

    def states(self):
        """Get state of each node.

        if agent is good:
            state contains node's value & node's weights & node's adja values.
        else:
            state contains other good agents' value.
        """
        states = [[] for _ in range(self.nodes_n)]
        for i, node in enumerate(self.nodes):
            if node.property == Property.GOOD:
                for v, _ in node.weights.items():
                    states[i].append(self.nodes[v].v)
            elif node.property == Property.RIVAL:
                for node in self.nodes:
                    if node.property == Property.GOOD:
                        states[i].append(node.v)
            elif node.property == Property.MADDPG:
                for v, w in node.weights.items():
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

    def node_val(self):
        res = {}
        for i, node in enumerate(self.nodes):
            res['Node{0}'.format(i)] = node.v
        return res


class Env:

    def __init__(self, nodes_n, evil_nodes_type, reset_env=True, times=1, with_noise=False, directed_graph=False):
        self.nodes_n = nodes_n
        self.times = times
        self.with_noise = with_noise
        self.goods_n = 7
        self.rivals_n = 1
        self.randoms_n = 0
        self.constants_n = 2
        self.reset_env = reset_env
        self.directed_graph = directed_graph
        self.distances = [0 for _ in range(self.nodes_n)]
        self.evil_nodes_type = evil_nodes_type
        self.map, self.features_n, self.outputs_n = self.make_map()

    def reset(self):
        """If we need long time exploit, we should reset state every episode beginning.
        """
        if self.reset_env:
            self.map, self.features_n, self.outputs_n = self.make_map()
        self.distances = [self.__calc_distance(i) for i in range(self.nodes_n)]
        return self.states()

    def make_map(self):
        """Deterministic strategy now.
        """
        node_random_1 = Node(0, random.random() * self.times, Property.RANDOM)
        node_random_2 = Node(1, random.random() * self.times, Property.RANDOM)
        node_random_3 = Node(2, random.random() * self.times, Property.RANDOM)
        node_constant_1 = Node(1, random.random() * self.times, Property.CONSTANT)
        node_constant_2 = Node(2, random.random() * self.times, Property.CONSTANT)
        node_constant_3 = Node(0, random.random() * self.times, Property.CONSTANT)
        node_rival_1 = Node(0, random.random() * self.times, Property.RIVAL)
        node_rival_2 = Node(1, random.random() * self.times, Property.RIVAL)
        node_rival_3 = Node(2, random.random() * self.times, Property.RIVAL)
        node_maddpg_0 = Node(0, random.random() * self.times, Property.MADDPG)
        node_maddpg_1 = Node(1, random.random() * self.times, Property.MADDPG)
        node_maddpg_2 = Node(2, random.random() * self.times, Property.MADDPG)
        if self.evil_nodes_type == '3r':
            evil_nodes = [node_random_1, node_random_2, node_random_3]
        elif self.evil_nodes_type == '2r1c':
            evil_nodes = [node_random_1, node_random_2, node_constant_2]
        elif self.evil_nodes_type == '1r2c':
            evil_nodes = [node_random_1, node_constant_1, node_constant_2]
        elif self.evil_nodes_type == '3c':
            evil_nodes = [node_constant_3, node_constant_1, node_constant_2]
        elif self.evil_nodes_type == 'maddpg':
            evil_nodes = [node_maddpg_0, node_maddpg_1, node_maddpg_2]
        elif self.evil_nodes_type == 'rival':
            evil_nodes = [node_rival_1, node_rival_2, node_rival_3]
        nodes = evil_nodes + \
                [Node(i, random.random() * self.times, Property.GOOD) for i in range(3, self.nodes_n)]
        if not self.directed_graph:
            # undirected graph
            print('Use Undirected Graph!')
            nodes[0].weights = {0: 1, 6: 0, 7: 0}
            nodes[1].weights = {1: 1, 3: 0, 7: 0, 8: 0}
            nodes[2].weights = {2: 1, 3: 0, 4: 0, 5: 0, 9: 0}
            nodes[3].weights = {1: 0.2, 3: 0.2, 5: 0.2, 6: 0.2, 9: 0.2}
            nodes[4].weights = {2: 0.2, 4: 0.2, 6: 0.2, 7: 0.2, 8: 0.2}
            nodes[5].weights = {2: 0.2, 3: 0.2, 5: 0.2, 7: 0.2, 8: 0.2}
            nodes[6].weights = {0: 0.2, 3: 0.2, 4: 0.2, 6: 0.2, 7: 0.2}
            nodes[7].weights = {0: 0.125, 1: 0.125, 4: 0.125, 5: 0.125, 6: 0.125, 7: 0.125, 8: 0.125, 9: 0.125}
            nodes[8].weights = {1: 0.2, 4: 0.2, 5: 0.2, 7: 0.2, 8: 0.2}
            nodes[9].weights = {2: 0.25, 3: 0.25, 7: 0.25, 9: 0.25}
        else:
            print('Use Directed Graph!')
            # direct graph
            nodes[0].weights = {0: 1}
            nodes[1].weights = {1: 1}
            nodes[2].weights = {2: 1}
            nodes[3].weights = {0: 2, 1: 0.2, 3: 0.2, 4: 0.2, 7: 0.2}
            nodes[4].weights = {2: 0.25, 4: 0.25, 5: 0.25, 8: 0.25}
            nodes[5].weights = {2: 0.33, 3: 0.33, 5: 0.33}
            nodes[6].weights = {1: 0.25, 4: 0.25, 5: 0.25, 6: 0.25}
            nodes[7].weights = {0: 0.25, 4: 0.25, 6: 0.25, 7: 0.25}
            nodes[8].weights = {1: 0.25, 3: 0.25, 5: 0.25, 8: 0.25}
            nodes[9].weights = {2: 0.25, 3: 0.25, 6: 0.25, 9: 0.25}
        features_n = []
        outputs_n = []
        for i, x in enumerate(nodes):
            if x.property == Property.GOOD:
                # features: self node and neighbors' node value
                # outputs: neighbors' node weight
                features_n.append(x.neighbors_n)
                outputs_n.append(x.neighbors_n - 1)
            elif x.property == Property.RIVAL:
                features_n.append(self.goods_n)
                # output self value
                outputs_n.append(1)
            elif x.property == Property.MADDPG:
                features_n.append(x.neighbors_n)
                # output self value
                outputs_n.append(1)
            else:
                # doesn't need to train
                features_n.append(-1)
                outputs_n.append(-1)
        return Map(nodes, times=self.times), features_n, outputs_n

    def step(self, action, node_i, is_continuous=False, update_value=False):
        """Update node_i's weights.
        """
        if update_value:
            self.map.update_value_directly(node_i, action.item())
        else:
            # update weight
            if is_continuous:
                # action = action[0]
                i = j = 0
                while i < len(action):
                    if node_i == self.map.weights_index[node_i][j]:
                        j += 1
                        continue
                    self.map.update_by_weight_index(node_i, j, action[i])
                    i += 1
                    j += 1
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
        r = self.__reward(node_i)
        return r

    def __reward(self, node_i):
        """Give agent reward.
        """
        r = 0
        property = self.map.nodes[node_i].property
        if property is Property.GOOD:
            d = 0
            for j, w in self.map.nodes[node_i].weights.items():
                d += abs(self.map.nodes[j].v - self.map.nodes[node_i].v) * w
            # ddpg 3r 
            # r = 1 * (math.exp(-20 * d) - 0.5)
            r = 1 * (math.exp(-20 * d) - 0.5)
        elif property is Property.MADDPG:
            d = 0
            t = 0
            for i in range(self.nodes_n-self.goods_n, self.nodes_n):
                for j in range(i, self.nodes_n):
                    d += abs(self.map.nodes[i].v - self.map.nodes[j].v)
                    t += 1
            d /= t
            r = 1 * (math.exp(-20 * d) - 0.5)
        elif property is Property.RIVAL:
            d = 0
            t = 0
            for i in range(self.nodes_n-self.goods_n, self.nodes_n):
                for j in range(i, self.nodes_n):
                    d += abs(self.map.nodes[i].v - self.map.nodes[j].v)
                    t += 1
            d /= t
            r = 1 * (math.log(20 * d + 1) - 0.5)
        return r

    def __calc_distance(self, node_i):
        """Calc distance of node_i with other connected nodes.
        """
        d = 0
        n = len(self.map.nodes[node_i].weights)
        for k, v in self.map.nodes[node_i].weights.items():
            d += (self.map.nodes[node_i].v - self.map.nodes[k].v) ** 2
        return math.sqrt(d / n)

    def update_value_of_node(self, rival_action_args=None):
        """When synchronize update node, use this function when
         all node's updating finished.
         When asynchronous update node, use this function when
         each node's updating finished.
        """
        self.map.update_value_of_node(with_noise=self.with_noise)
        if rival_action_args and type(rival_action_args) is dict:
            for node_i, action in rival_action_args.items():
                self.map.nodes[node_i].update_v(action)

    def states(self):
        """Get current states.
        """
        return self.map.states()

    def is_done(self, tolerance=0.1):
        """Check each two node's value distance lower than a mini number.
        """
        start = self.nodes_n - self.goods_n
        for i in range(start, self.nodes_n):
            for j in range(i+1, self.nodes_n):
                if abs(self.map.nodes[i].v - self.map.nodes[j].v) > tolerance:
                    return False
        return True

    def is_good(self, node_i):
        """Return if agent node_i is GOOD agent.
        """
        return self.map.nodes[node_i].property == Property.GOOD

    def is_rival(self, node_i):
        """Return if agent node_i is RIVAL agent.
        """
        return self.map.nodes[node_i].property == Property.RIVAL
