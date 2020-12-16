import random
from env import Env
import math
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, v in node.weights.items():
            q[i][k] = init_value
    return q


def generate_good_and_bad_set(env: Env):
    goods, bads = [], []
    for i, node in enumerate(env.map.nodes):
        node_good = []
        node_bad = []
        for k, _ in node.weights.items():
            if k >= env.nodes_n - env.goods_n and k != i:
                node_good.append(k)
            else:
                node_bad.append(k)
        goods.append(node_good)
        bads.append(node_bad)
    return goods, bads


def find_choosen(i, goods_of_i, bads_of_i):
    if i < 3:
        return []
    l_g, l_b = len(goods_of_i), len(bads_of_i)
    num_g, num_b = np.random.randint(1, l_g + 1), np.random.randint(0, l_b + 1)
    choosen_g = random.sample(goods_of_i, num_g)
    choosen_b = random.sample(bads_of_i, num_b)
    choosen = choosen_g + choosen_b
    if i not in choosen:
        choosen.append(i)
    return choosen


def q_consensus(**kwargs):
    # print(kwargs)
    writer = SummaryWriter("logs/switch_q_consensus")
    env = Env(
        nodes_n=10,
        reset_env=kwargs["reset_env"],
        evil_nodes_type=kwargs["evil_nodes_type"],
        times=1,
        with_noise=kwargs["with_noise"],
        directed_graph=kwargs["directed_graph"],
    )
    Q = make_q(env, 1)
    goods, bads = generate_good_and_bad_set(env)
    step_size = 0.01
    if kwargs["save_csv"]:
        df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.nodes_n)])
    episodes_n = 3000
    r = [[None for _ in range(env.nodes_n)] for _ in range(env.nodes_n)]
    for epi in range(episodes_n):
        if kwargs["save_csv"]:
            df = df.append(env.map.node_val(), ignore_index=True)
        for i, node in enumerate(env.map.nodes):
            choosen = find_choosen(i, goods[i], bads[i])
            for j, q in Q[i].items():
                # 0.01 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                # times=1, directed graph: 1 + 0.1 * epi, range(2000)
                # times=1, undirected graph: 10 + 0.1 * epi, range(1000)
                if j in choosen:
                    r[i][j] = math.exp(-abs(env.map.nodes[j].v - node.v) * 100)
                if epi > 0.8 * episodes_n:
                    step_size -= 0.01 / (episodes_n * 0.2)
                if r[i][j]:
                    Q[i][j] += max(step_size, 0) * (r[i][j] - Q[i][j])
            # update q_sum
            q_sum = 0
            for k in choosen:
                q_sum = Q[i][k]
            # update action
            for k in choosen:
                w = (Q[i][k] / q_sum) * (1 - 1 / len(Q[i]))
                env.map.update_by_weight(i, k, w)
            # normalize weights
            env.map.normalize(i, choosen)

            # writer.add_scalars(
            #     "Q of Node_{0}".format(i),
            #     {"Adj_{0}".format(k): Q[i][k] for k in Q[i].keys()},
            #     epi,
            # )
            # writer.add_scalars(
            #     "R of Node_{0}".format(i),
            #     {
            #         "Adj_{0}".format(k): r[i][k] if r[i][k] is not None else 0
            #         for k in Q[i].keys()
            #     },
            #     epi,
            # )
        env.map.update_value_of_node(with_noise=kwargs["with_noise"])
    print(env.map)
    with_noise_str = "" if not kwargs["with_noise"] else "noise_"
    directed_str = "" if not kwargs["directed_graph"] else "directed_"
    if kwargs["save_csv"]:
        df.to_csv(
            "q_new_{0}{1}{2}.csv".format(
                with_noise_str, directed_str, kwargs["evil_nodes_type"]
            )
        )
    return env


if __name__ == "__main__":
    q_consensus()
