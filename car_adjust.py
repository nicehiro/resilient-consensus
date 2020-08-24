from env import Env, Node, Property, Map
import random
import pandas as pd
import math


class CarEnv(Env):
    def __init__(self):
        super().__init__(nodes_n=6, evil_nodes_type='3r', times=1, reset_env=True, with_noise=False, directed_graph=False)

    def make_map(self):
        node_random_1 = Node(0, random.random() * self.times, Property.RANDOM)
        node_random_2 = Node(1, random.random() * self.times, Property.RANDOM)
        nodes = [node_random_1, node_random_2] + \
                [Node(i, random.random() * self.times, Property.GOOD) for i in range(2, self.nodes_n)]
        nodes[0].weights = {0: 1, 3: 0, 5: 0}
        nodes[1].weights = {1: 1, 2: 0, 3: 0, 4: 0}
        nodes[2].weights = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        nodes[3].weights = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        nodes[4].weights = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        nodes[5].weights = {0: 0.25, 2: 0.25, 4: 0.25, 5: 0.25}
        features_n = []
        outputs_n = []
        for i, x in enumerate(nodes):
            if x.property == Property.GOOD:
                # features: self node and neighbors' node value
                # outputs: neighbors' node weight
                features_n.append(x.neighbors_n)
                outputs_n.append(x.neighbors_n - 1)
            else:
                # doesn't need to train
                features_n.append(-1)
                outputs_n.append(-1)
        return Map(nodes, times=self.times), features_n, outputs_n



def make_q(env: Env, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, v in node.weights.items():
            q[i][k] = init_value
    return q


def q_consensus(**kwargs):
    # print(kwargs)
    env = CarEnv()
    Q = make_q(env, 1)
    step_size = 0.01
    if kwargs['save_csv']:
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(env.nodes_n)])
    for epi in range(1, 1001):
        if kwargs['save_csv']:
            df = df.append(env.map.node_val(), ignore_index=True)
        for i, node in enumerate(env.map.nodes):
            for j, q in Q[i].items():
                # 0.01 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                # times=1, directed graph: 1 + 0.1 * epi, range(2000)
                # times=1, undirected graph: 10 + 0.1 * epi, range(1000)
                r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * (10 + 0.1 * epi))
                Q[i][j] += max((step_size), 0) * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i])) + 0.001
                env.map.update_by_weight(i, j, w)
            env.map.normalize(i)
        env.map.update_value_of_node(with_noise=kwargs['with_noise'])
    print(env.map)
    with_noise_str = '' if not kwargs['with_noise'] else 'noise_'
    directed_str = '' if not kwargs['directed_graph'] else 'directed_'
    if kwargs['save_csv']:
        df.to_csv('car_{0}{1}{2}.csv'.format(with_noise_str, directed_str, '3r'))
    return env


if __name__ == '__main__':
    q_consensus(save_csv=True, with_noise=False, directed_graph=False)