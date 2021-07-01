from typing import List
from rcenv import Env
from modified_rl.agent import Agent
from attribute import Attribute
from utils import adjacent_matrix
import random
import pandas as pd
import numpy as np
import torch


def test(**kwargs):
    # load model
    bad_node_attrs = {
        "rrrr": [Attribute.RANDOM] * 4,
        "rrcc": [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        "cccc": [Attribute.CONSTANT] * 4,
    }
    node_attrs = bad_node_attrs[kwargs["bad_attrs"]] + [Attribute.NORMAL] * 8
    probs = kwargs["probs"]
    seeds = [random.random() * 100 for _ in range(12)]
    env = Env(
        adjacent_matrix, node_attrs, probs, seeds, noise_scale=kwargs["noise_scale"]
    )

    bads_n = 4
    goods_n = 8

    normals = [
        Agent(
            obs_dim=env.features_n[i + bads_n]-1,
            act_dim=env.actions_n[i + bads_n],
            buffer_size=kwargs["memory_size"],
            lr=kwargs["actor_lr"],
            gamma=0.95,
            node_i=i + bads_n,
            restore_path=kwargs["restore_path"],
            batch_size=kwargs["batch_size"],
            value=env.topology.nodes[i + bads_n].value,
            node_index_of_weights=0,
        )
        for i in range(goods_n)
    ]

    for agent in normals:
        agent.restore()

    episodes_n = kwargs["episodes_n"]

    df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.n)])

    # update topology
    o = env.reset()
    acts = [None for _ in range(bads_n)] + [
        np.full(
            (1, len(env.topology.nodes[i + bads_n].adjacents)),
            1 / len(env.topology.nodes[i + bads_n].adjacents),
        ).squeeze()
        for i in range(goods_n)
    ]
    for episode in range(episodes_n):
        res = {}
        for i, node in enumerate(env.topology.nodes):
            if type(node.value) is torch.Tensor:
                res["Node{0}".format(i)] = node.value.item()
            else:
                res["Node{0}".format(i)] = node.value

        df = df.append(res, ignore_index=True)
        for i, normal in enumerate(normals):
            a = normal.act(o[i + bads_n], acts[i + bads_n])
            acts[i + bads_n] = a
        rews, o_ = env.step(acts)
        for i, normal in enumerate(normals):
            normal.update_value(env.topology.nodes[i + bads_n].value)
        o = o_

    prefix = ""
    for attr in bad_node_attrs[kwargs["bad_attrs"]][0:4]:
        if attr is Attribute.RANDOM:
            prefix += "r"
        if attr is Attribute.CONSTANT:
            prefix += "c"
    df.to_csv("rl_{0}_{1}.csv".format(prefix, int(kwargs["probs"][0] * 10)))


if __name__ == "__main__":
    probs = [1.0] * 4 + [1.0] * 8
    test(
        noise_scale=0.01,
        probs=probs,
        bad_attrs="cccc",
        log_path="logs/modified_rl/",
        memory_size=1000,
        actor_lr=1e-3,
        restore_path="trained/cccc-10/",
        batch_size=640,
        episodes_n=50,
    )