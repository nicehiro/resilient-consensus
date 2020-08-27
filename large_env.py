from random import random
import math

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from env import Map, Node, Property


class LargeNet:
    def __init__(self) -> None:
        self.nodes_n = 1000
        self.bads_n = 100
        self.connected_prob = 0.01
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
        for node in good_nodes:
            for i in range(self.nodes_n):
                prob = random()
                if prob < self.connected_prob:
                    node.weights[i] = 1
            node.weights[node.index] = 1
            for k in node.weights.keys():
                node.weights[k] = 1 / len(node.weights)
        self.map = Map(nodes)

    def verify_connected(self):
        visited = {self.bads_n}
        s = [self.bads_n]
        while len(s) > 0:
            f = self.map.nodes[s.pop(0)]
            for k in f.weights.keys():
                if k in visited:
                    continue
                visited.add(k)
                s.append(k)
        return len(visited) == self.nodes_n

    def is_done(self, tolerance=0.1):
        for i in range(self.bads_n, self.nodes_n):
            for j in range(i+1, self.nodes_n):
                if abs(self.map.nodes[i].v - self.map.nodes[j].v) > tolerance:
                    return False
        return True

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
    print(env.verify_connected())
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
    q_consensus(save_csv=False, directed_graph=False, with_noise=False,
                log_path='larget_net_logs')
