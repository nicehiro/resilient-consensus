import enum
from random import random
import math

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from env import Map, Node, Property


class LargeNet:
    def __init__(self) -> None:
        self.nodes_n = 1000
        self.bads_n = 100
        self.goods_n = 900
        self.connected_prob = 0.001
        self.map = None
        self.make_map()

    def make_map(self):
        bad_nodes = [Node(i, random(), Property.CREEPY)
                        for i in range(self.bads_n)]
        good_nodes = [Node(i, random(), Property.GOOD)
                        for i in range(self.bads_n, self.nodes_n)]
        nodes = bad_nodes + good_nodes
        for i, node in enumerate(bad_nodes):
            node.weights = {i: 1.0}
        for i in range(1, 10):
            nodes[i * 100].weights = {i * 100 + 1: 1.0}
            for j in range(i*100, i*100+99):
                nodes[j].weights[j+1] = 1.0
                nodes[j+1].weights = {j: 1.0}
            a, b = random(), random()
            if i < 9:
                nodes[math.floor(a * 100 + i * 100)].weights[math.floor(b * 100 + (i + 1) * 100)] = 1.0
        for i in range(100):
            for j in range(100, 1000):
                if random() < self.connected_prob:
                    nodes[j].weights[i] = 1.0
        self.map = Map(nodes)
        

    # def make_map(self):
    #     bad_nodes = [Node(i, random(), Property.CREEPY)
    #                     for i in range(self.bads_n)]
    #     good_nodes = [Node(i, random(), Property.GOOD)
    #                     for i in range(self.bads_n, self.nodes_n)]
    #     nodes = bad_nodes + good_nodes
    #     connected = set()
    #     not_connected = set()
    #     for i, node in enumerate(bad_nodes):
    #         node.weights = {i: 1.0}
    #     self.make_connections(good_nodes)
    #     self.map = Map(nodes)
    #     while not self.verify_connected():
    #         # for j, node in enumerate(good_nodes):
    #         #     if j in connected:
    #         #         continue
    #         #     for i in range(self.nodes_n):
    #         #         prob = random()
    #         #         if prob < self.connected_prob:
    #         #             node.weights[i] = 1
    #         #             connected.add(i)
    #         #             connected.add(j)
    #         #     node.weights[node.index] = 1
    #         #     for k in node.weights.keys():
    #         #         node.weights[k] = 1 / len(node.weights)
    #         self.make_connections(good_nodes)
    #         self.map = Map(nodes)

    def find_max_connections(self):
        max_c, min_c, mean = 0, 100, 0
        for node in self.map.nodes[self.bads_n:]:
            max_c = max(max_c, len(node.weights))
            min_c = min(min_c, len(node.weights))
            mean += len(node.weights)
        return max_c, min_c, mean / 900


    def make_connections(self, nodes):
        for node in nodes:
            for i in range(self.nodes_n):
                prob = random()
                if prob < self.connected_prob:
                    node.weights[i] = 1
            node.weights[node.index] = 1
            for k in node.weights.keys():
                node.weights[k] = 1 / len(node.weights)

    def verify_connected(self):
        visited = {self.bads_n}
        s = [self.bads_n]
        while len(s) > 0:
            f = self.map.nodes[s.pop(0)]
            for k in f.weights.keys():
                if k in visited or k < 100:
                    continue
                visited.add(k)
                s.append(k)
        return len(visited) == self.goods_n

    def is_done(self, tolerance=0.1):
        for i in range(self.bads_n, self.nodes_n):
            for j in range(i+1, self.nodes_n):
                if abs(self.map.nodes[i].v - self.map.nodes[j].v) > tolerance:
                    return False
        return True


def generate_connected_net():
    for i in range(1000):
        env = LargeNet()
        if env.verify_connected():
            print(True)
            print(env.find_max_connections())
            return True
        else:
            print(False)


def make_q(env: LargeNet, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, _ in node.weights.items():
            q[i][k] = init_value
    return q


def q_consensus(**kwargs):
    # print(kwargs)
    writer = SummaryWriter(kwargs['log_path'])
    env = LargeNet()
    print(env.find_max_connections())
    Q = make_q(env, 1)
    step_size = 0.01
    if kwargs['save_csv']:
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(env.nodes_n)])
    episodes_n = 1000
    observe_nodes = [int(random() * 1000) for _ in range(10)]
    for epi in range(episodes_n):
        if kwargs['save_csv']:
            df = df.append(env.map.node_val(), ignore_index=True)
        writer.add_scalars('Values', {'Node {0}'.format(i): env.map.nodes[i].v for i in observe_nodes}, epi)
        loss = 0
        for i in range(env.bads_n, env.nodes_n-1):
            loss += (env.map.nodes[i].v - env.map.nodes[i+1].v) ** 2
        writer.add_scalar('Loss', math.sqrt(loss), epi)
        for i, node in enumerate(env.map.nodes):
            for j, q in Q[i].items():
                # 0.01 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                # times=1, directed graph: 1 + 0.1 * epi, range(2000)
                # times=1, undirected graph: 10 + 0.1 * epi, range(1000)
                r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * (1 + 0.1 * epi))
                if epi > 0.8 * episodes_n:
                    step_size -= 0.01 / (episodes_n * 0.2)
                Q[i][j] += max(step_size, 0) * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i]))
                env.map.update_by_weight(i, j, w)
            env.map.normalize(i)
        env.map.update_value_of_node(with_noise=kwargs['with_noise'])
    print(env.is_done())
    with_noise_str = '' if not kwargs['with_noise'] else 'noise_'
    directed_str = '' if not kwargs['directed_graph'] else 'directed_'
    if kwargs['save_csv']:
        df.to_csv('q_new_{0}{1}{2}.csv'.format(with_noise_str, directed_str, kwargs['evil_nodes_type']))
    return env


if __name__ == '__main__':
    # q_consensus(save_csv=False, directed_graph=False, with_noise=False,
    #             log_path='large_net_logs')
    generate_connected_net()
