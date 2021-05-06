import random
from rcenv import Env
import math
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from attribute import Attribute
from functools import reduce


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.n)]
    for i, node in enumerate(env.topology.nodes):
        for adj, _ in node.weights.items():
            q[i][adj] = init_value
    return q


def generate_good_and_bad_set(env: Env):
    goods, bads = [], []
    for i, node in enumerate(env.topology.nodes):
        node_good = []
        node_bad = []
        for adj, _ in node.weights.items():
            if node != adj and adj.attribute is Attribute.NORMAL:
                node_good.append(adj)
            elif node != adj and adj.attribute is not Attribute.NORMAL:
                node_bad.append(adj)
        goods.append(node_good)
        bads.append(node_bad)
    return goods, bads


def find_choosen(node_i, goods_of_i, bads_of_i, connect_probs):
    if node_i.attribute is not Attribute.NORMAL:
        return []
    l_g, l_b = len(goods_of_i), len(bads_of_i)
    choosen_g, choosen_b = [], []
    for i in goods_of_i:
        if random.random() < connect_probs:
            choosen_g.append(i)
    for i in bads_of_i:
        if random.random() < connect_probs:
            choosen_b.append(i)
    choosen = choosen_g + choosen_b
    choosen.append(node_i)
    return choosen


def q_consensus(**kwargs):
    # print(kwargs)
    writer = SummaryWriter("logs/switch_q_consensus")
    seed = 1
    adj_matrix = [[1] * 12 for _ in range(12)]
    bads_attrs = {
        "rrrr": [Attribute.RANDOM] * 4,
        "rrcc": [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        "ccrr": [Attribute.CONSTANT] * 2 + [Attribute.RANDOM] * 2,
        "cccc": [Attribute.CONSTANT] * 4,
    }
    node_attrs = bads_attrs[kwargs["bads_attrs"]] + [Attribute.NORMAL] * 8
    # node_attrs = [Attribute.NORMAL] * 12
    probs = [1.0] * 12
    env = Env(
        adj_matrix=adj_matrix,
        node_attrs=node_attrs,
        times=1,
        probs=probs,
        seeds=kwargs["seeds"],
        noise_scale=kwargs["noise_scale"],
    )
    Q = make_q(env, 1)
    goods, bads = generate_good_and_bad_set(env)
    step_size = 0.1
    df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.n)])
    success = []
    episodes_n = kwargs["episodes_n"]
    r = [{} for _ in range(env.n)]
    epi = 0
    for epi in range(episodes_n):
        if epi > 0.8 * episodes_n:
            step_size -= 0.1 / (episodes_n * 0.2)
        df = df.append(env.topology.node_val(), ignore_index=True)
        for i, node in enumerate(env.topology.nodes):
            if i < 4:
                continue
            choosen = find_choosen(node, goods[i], bads[i], kwargs["connect_probs"])
            # update weights
            q_sum = 0
            for k in choosen:
                q_sum += Q[i][k]
            # update action
            for k, v in node.weights.items():
                if k in choosen:
                    w = Q[i][k] / q_sum
                else:
                    w = 0
                node.update_weight_by_adj(k, w)

            rewards = {}
            for j, q in Q[i].items():
                if j in choosen:
                    noise = kwargs['noise_scale'] * (random.random() * 2 - 1)
                    r[i][j] = math.exp(-1 * abs(j.value - node.value + noise))
                    r[i][j] = node.weights[j] * r[i][j]
                    rewards[j] = r[i][j]
            sum_r = sum(rewards.values())
            for j, q in Q[i].items():
                if j in choosen:
                    if j.index == i:
                        cc = random.random()
                        if cc > kwargs['connect_probs']:
                            continue
                    r_ij = rewards[j] / sum_r
                    Q[i][j] += max(step_size, 0) * (r_ij - Q[i][j])

            # update weights
            q_sum = 0
            for k in choosen:
                q_sum += Q[i][k]
            # update action
            for k, v in node.weights.items():
                if k in choosen:
                    w = Q[i][k] / q_sum
                else:
                    w = 0
                node.update_weight_by_adj(k, w)
            # for j, q in Q[i].items():
            #     if j in choosen:
            #         r[i][j] = math.exp(-abs(j.value - node.value) * (10 + 0.1 * epi))
            #         Q[i][j] += max(step_size, 0) * (r[i][j] - Q[i][j])
            # update q_sum

        # check success
        t = df.iloc[-1:, 5:]
        if (t.max(axis=1) - t.min(axis=1)).iloc[-1] < kwargs["baseline"]:
            success.append(True)
        else:
            success.append(False)
        if kwargs["check_success"] and epi > 10:
            # calc expectation of distance in last 10 steps
            last_steps = 20
            is_consensus = reduce(lambda x, y: x and y, success[-last_steps:])
            if is_consensus:
                break
        env.topology.update_value()
    print(env.topology)
    if kwargs["save_csv"]:
        df.to_csv(
            "q_c_switcher_{0}_{1}.csv".format(
                kwargs["bads_attrs"], int(kwargs["connect_probs"] * 10)
            )
        )
    # print(epi)
    return epi, is_consensus


if __name__ == "__main__":
    # cp=0.9, baseline=0.02, epi=180
    q_consensus(
        noise_scale=0.01,
        seeds=[i for i in range(12)],
        save_csv=True,
        episodes_n=3000,
        bads_attrs="rrrr",
        connect_probs=0.1,
        check_success=True,
        baseline=0.04,
    )
