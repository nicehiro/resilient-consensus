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
    env = Env(nodes_n=10, reset_env=kwargs['reset_env'], evil_nodes_type=kwargs['evil_nodes_type'], times=1000, with_noise=kwargs['with_noise'])
    Q = make_q(env, 1)
    step_size = 0.01
    if kwargs['save_csv']:
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(env.nodes_n)])
    for epi in range(1000):
        if kwargs['save_csv']:
            df = df.append(env.map.node_val(), ignore_index=True)
        for i, node in enumerate(env.map.nodes):
            for j, q in Q[i].items():
                # 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * (0.01 + 0.0001 * epi))
                Q[i][j] += step_size * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i])) + 0.001
                env.map.update_by_weight(i, j, w)
            env.map.normalize(i)
        env.map.update_value_of_node(with_noise=kwargs['with_noise'])
    # print(env.map)
    with_noise_str = '' if not kwargs['with_noise'] else 'noise_'
    if kwargs['save_csv']:
        df.to_csv('q_new_{0}{1}.csv'.format(with_noise_str, kwargs['evil_nodes_type']))
    return env


if __name__ == '__main__':
    q_consensus()
