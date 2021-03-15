from env import Env
import math
import pandas as pd
import numpy as np


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, v in node.weights.items():
            q[i][k] = init_value
    return q


def q_consensus(**kwargs):
    # print(kwargs)
    env = Env(nodes_n=10, reset_env=kwargs['reset_env'], evil_nodes_type=kwargs['evil_nodes_type'], times=1, with_noise=kwargs['with_noise'], directed_graph=kwargs['directed_graph'])
    Q = make_q(env, 1)
    step_size = 0.01
    if kwargs['save_csv']:
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(env.nodes_n)])
    episodes_n = 5000
    for epi in range(episodes_n):
        if kwargs['save_csv']:
            df = df.append(env.map.node_val(), ignore_index=True)
        switch = []
        for i, node in enumerate(env.map.nodes):
            connected = []
            for j, q in Q[i].items():
                if np.random.random() > 0.5:
                    continue
                connected.append(j)
                # 0.01 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                # times=1, directed graph: 1 + 0.1 * epi, range(2000)
                # times=1, undirected graph: 10 + 0.1 * epi, range(1000)
                r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * (10 + 0.01 * epi))
                if epi > 0.8 * episodes_n:
                    step_size -= 0.01 / (episodes_n * 0.2)
                Q[i][j] += max(step_size, 0) * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                if j not in connected:
                    continue
                w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i])) + 0.001
                env.map.update_by_weight(i, j, w)
            env.map.normalize(i)
            switch.append(connected)
        env.map.update_value_of_node(with_noise=kwargs['with_noise'], switch=switch)
    print(env.map)
    with_noise_str = '' if not kwargs['with_noise'] else 'noise_'
    directed_str = '' if not kwargs['directed_graph'] else 'directed_'
    if kwargs['save_csv']:
        df.to_csv('q_new_switch_{0}{1}{2}.csv'.format(with_noise_str, directed_str, kwargs['evil_nodes_type']))
    return env


if __name__ == '__main__':
    q_consensus(reset_env=True, evil_nodes_type='3c', times=1, with_noise=True, directed_graph=True, save_csv=True)
