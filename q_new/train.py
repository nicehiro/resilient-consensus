from rcenv import Env
import math
import pandas as pd
import numpy as np
from utils import adjacent_matrix
from attribute import Attribute


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.n)]
    for i, node in enumerate(env.topology.nodes):
        for adj, _ in node.weights.items():
            if adj.index == i:
                continue
            q[i][adj.index] = init_value
    return q


def q_consensus(**kwargs):
    # print(kwargs)
    node_attrs = [
        Attribute.CONSTANT,
        Attribute.CONSTANT,
        Attribute.RANDOM,
        Attribute.RANDOM,
    ] + [Attribute.NORMAL] * 8
    env = Env(adjacent_matrix, node_attrs)
    Q = make_q(env, 1)
    step_size = 0.01
    if kwargs["save_csv"]:
        df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.n)])
    episodes_n = 2000
    for epi in range(episodes_n):
        if kwargs["save_csv"]:
            df = df.append(env.topology.node_val(), ignore_index=True)
        for i, node in enumerate(env.topology.nodes):
            for j, q in Q[i].items():
                # 0.01 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                # times=1, directed graph: 1 + 0.1 * epi, range(2000)
                # times=1, undirected graph: 10 + 0.1 * epi, range(1000)
                r_ij = math.exp(
                    -abs(env.topology.nodes[j].value - node.value) * (1 + 0.1 * epi)
                )
                if epi > 0.8 * episodes_n:
                    step_size -= 0.01 / (episodes_n * 0.2)
                Q[i][j] += max(step_size, 0) * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                w = (Q[i][j] / q_sum) * (1 - 1 / len(node.weights))
                node.update_weight_by_adj(env.topology.nodes[j], w)
        env.topology.update_value()
    print(env.topology)
    prefix = ""
    for attr in node_attrs[0:4]:
        if attr is Attribute.RANDOM:
            prefix += "r"
        if attr is Attribute.CONSTANT:
            prefix += "c"
    if kwargs["save_csv"]:
        df.to_csv("q_c_{0}.csv".format(prefix))
    return env


if __name__ == "__main__":
    q_consensus()
