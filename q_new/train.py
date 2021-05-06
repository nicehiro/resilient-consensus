from rcenv import Env
import math
import pandas as pd
import numpy as np
from utils import adjacent_matrix
from attribute import Attribute
from functools import reduce
import random


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.n)]
    for i, node in enumerate(env.topology.nodes):
        for adj, _ in node.weights.items():
            q[i][adj.index] = init_value
    return q


def q_consensus(**kwargs):
    # print(kwargs)
    bads_attrs = {
        "rrrr": [Attribute.RANDOM] * 4,
        "rrcc": [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        "ccrr": [Attribute.CONSTANT] * 2 + [Attribute.RANDOM] * 2,
        "cccc": [Attribute.CONSTANT] * 4,
    }
    node_attrs = bads_attrs[kwargs["bads_attrs"]] + [Attribute.NORMAL] * 8
    env = Env(
        adjacent_matrix,
        node_attrs=node_attrs,
        probs=kwargs["probs"],
        seeds=kwargs["seeds"],
        times=1,
        noise_scale=kwargs["noise_scale"],
    )
    is_consensus = False
    Q = make_q(env, 1)
    step_size = 0.1
    df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.n)])
    success = []
    episodes_n = kwargs["episodes_n"]
    for epi in range(episodes_n):
        if epi > 0.8 * episodes_n:
            step_size -= 0.1 / (episodes_n * 0.2)
        df = df.append(env.topology.node_val(), ignore_index=True)
        for i, node in enumerate(env.topology.nodes):
            if i < 4:
                continue
            rewards = {}
            for j, q in Q[i].items():
                node_j = env.topology.nodes[j]
                noise = kwargs['noise_scale'] * (random.random() * 2 - 1)
                r_ij = math.exp(-1 * abs(node_j.value - node.value + noise))
                r_ij = node.weights[node_j] * r_ij
                rewards[j] = r_ij

            sum_r = sum(rewards.values())
            for j, q in Q[i].items():
                r_ij = rewards[j] / sum_r
                Q[i][j] += max(step_size, 0) * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                w = Q[i][j] / q_sum
                node.update_weight_by_adj(env.topology.nodes[j], w)
        # check success
        t = df.iloc[-1:, 5:]
        if (t.max(axis=1) - t.min(axis=1)).iloc[-1] < kwargs["baseline"]:
            success.append(True)
        else:
            success.append(False)
        if kwargs["check_success"] and epi > 20:
            # calc expectation of distance in last 10 steps
            last_steps = 20
            is_consensus = reduce(lambda x, y: x and y, success[-last_steps:])
            if is_consensus:
                break
        env.topology.update_value()

    prefix = ""
    for attr in bads_attrs[kwargs["bads_attrs"]][0:4]:
        if attr is Attribute.RANDOM:
            prefix += "r"
        if attr is Attribute.CONSTANT:
            prefix += "c"
    if kwargs["save_csv"]:
        df.to_csv("q_c_{0}_{1}.csv".format(prefix, int(kwargs["probs"][0] * 10)))
    print(env.topology)
    return epi, is_consensus


if __name__ == "__main__":
    probs = [0.5] * 4 + [1.0] * 8
    q_consensus(
        probs=probs,
        noise_scale=0.05,
        seeds=[i for i in range(12)],
        save_csv=True,
        episodes_n=3000,
        bads_attrs="cccc",
        check_success=True,
        baseline=0.1,
    )
