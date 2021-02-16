import random
from rcenv import Env
import math
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from attribute import Attribute
from utils import adjacent_matrix


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


def find_choosen(node_i, goods_of_i, bads_of_i):
    if node_i.attribute is not Attribute.NORMAL:
        return []
    l_g, l_b = len(goods_of_i), len(bads_of_i)
    num_g, num_b = np.random.randint(1, l_g + 1), np.random.randint(1, l_b + 1)
    choosen_g = random.sample(goods_of_i, num_g)
    choosen_b = random.sample(bads_of_i, num_b)
    choosen = choosen_g + choosen_b
    choosen.append(node_i)
    return choosen


def q_consensus(**kwargs):
    # print(kwargs)
    writer = SummaryWriter("logs/switch_q_consensus")
    bads_attrs = {'rrrr': [Attribute.RANDOM] * 4,
                  'rrcc': [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
                  'ccrr': [Attribute.CONSTANT] * 2 + [Attribute.RANDOM] * 2,
                  'cccc': [Attribute.CONSTANT] * 4
                  }
    node_attrs = bads_attrs[kwargs['bads_attrs']] + [Attribute.NORMAL] * 8
    env = Env(
        adj_matrix=adjacent_matrix, node_attrs=node_attrs, times=1, has_noise=True
    )
    Q = make_q(env, 1)
    goods, bads = generate_good_and_bad_set(env)
    step_size = 0.01
    if kwargs["save_csv"]:
        df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.n)])
    episodes_n = kwargs['episodes_n']
    r = [{} for _ in range(env.n)]
    for epi in range(episodes_n):
        if kwargs["save_csv"]:
            df = df.append(env.topology.node_val(), ignore_index=True)
        for i, node in enumerate(env.topology.nodes):
            choosen = find_choosen(node, goods[i], bads[i])
            for j, q in Q[i].items():
                if epi > 0.8 * episodes_n:
                    step_size -= 0.01 / (episodes_n * 0.2)
                if j in choosen:
                    r[i][j] = math.exp(-abs(j.value - node.value) * 100)
                    Q[i][j] += max(step_size, 0) * (r[i][j] - Q[i][j])
            # update q_sum
            q_sum = 0
            for k in choosen:
                q_sum += Q[i][k]
            # update action
            weight_sum = 0
            for k, v in node.weights.items():
                if k in choosen:
                    w = (Q[i][k] / q_sum) * (1 - 1 / len(Q[i]))
                else:
                    w = 0
                weight_sum += w
                if k is not node:
                    node.update_weight_by_adj(k, w)
            node.normalize(weight_sum)
        env.topology.update_value()
    print(env.topology)
    if kwargs["save_csv"]:
        df.to_csv("q-consensus-switcher-{0}.csv".format(kwargs['bads_attrs']))
    return env


if __name__ == "__main__":
    q_consensus(save_csv=True, episodes_n=3000, bads_attrs='cccc')
