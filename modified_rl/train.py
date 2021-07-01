from rcenv import Env
from torch.utils.tensorboard import SummaryWriter
from modified_rl.agent import Agent
import numpy as np
from attribute import Attribute
from utils import adjacent_matrix
import random


def train(**kwargs):
    """Using modified to train a model make bad node's weight smaller enough."""
    writer = SummaryWriter(log_dir=kwargs["log_path"])
    bad_node_attrs = {
        "rrrr": [Attribute.RANDOM] * 4,
        "rrcc": [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        "cccc": [Attribute.CONSTANT] * 4,
    }
    node_attrs = bad_node_attrs[kwargs["bad_attrs"]] + [Attribute.NORMAL] * 8
    probs = kwargs["probs"]
    seeds = [random.random() * 100 for _ in range(12)]
    env = Env(
        adjacent_matrix,
        node_attrs,
        probs=probs,
        seeds=seeds,
        noise_scale=kwargs["noise_scale"],
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

    epochs_n = kwargs["epochs_n"]
    episodes_n = kwargs["episodes_n"]
    update_after = kwargs["update_after"]
    update_every = kwargs["update_every"]

    for epoch in range(epochs_n):
        # sample init value of x
        o = env.reset()
        # acts = [None for _ in range(bads_n)] + [
        #     np.full(
        #         (1, len(env.topology.nodes[i + bads_n].adjacents)),
        #         1 / len(env.topology.nodes[i + bads_n].adjacents),
        #     ).squeeze()
        #     for i in range(goods_n)
        # ]
        acts = [None] * bads_n + [
            np.random.dirichlet(np.ones(len(env.topology.nodes[i + bads_n].adjacents)))
            for i in range(goods_n)
        ]
        for episode in range(episodes_n):
            for i, normal in enumerate(normals):
                a = normal.act(o[i + bads_n], acts[i + bads_n])
                acts[i + bads_n] = a
            rews, o_ = env.step(acts)
            # update value
            for i, normal in enumerate(normals):
                normal.update_value(env.topology.nodes[i + bads_n].value)
            d = False
            # save to replay buffer
            for i, normal in enumerate(normals):
                normal.memory.store(
                    o[i + bads_n], acts[i + bads_n], rews[i], o_[i + bads_n], d
                )
            o = o_
            # record a to tensorboard
            for i, node in enumerate(env.topology.nodes):
                if node.attribute is Attribute.NORMAL:
                    writer.add_scalars(
                        "Actions of Node {0}".format(i),
                        {
                            "Adj {0}".format(adj.index): w
                            for adj, w in node.weights.items()
                        },
                        epoch * episodes_n + episode,
                    )
        # update policy
        if epoch % update_every == 0:
            loss = []
            for normal in normals:
                l = normal.optimize()
                loss.append(l)
                normal.memory.reset()
            # record loss to tensorboard
            writer.add_scalars(
                "Loss of Nodes",
                {"Node {0}".format(i): l for i, l in enumerate(loss)},
                epoch,
            )
        if epoch % 10 == 0:
            for agent in normals:
                agent.save()


if __name__ == "__main__":
    probs = [0.5] * 4 + [1.0] * 8
    train(
        bad_attrs="cccc",
        probs=probs,
        noise_scale=0.05,
        log_path="logs/modified_rl/",
        memory_size=1000,
        actor_lr=1e-3,
        restore_path="trained/",
        batch_size=640,
        epochs_n=4000,
        episodes_n=50,
        update_after=10,
        update_every=10,
    )
