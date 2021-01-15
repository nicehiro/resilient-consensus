from env import Property
from rcenv import Env
from torch.utils.tensorboard import SummaryWriter
from modified_rl.agent import Agent
import numpy as np
from attribute import Attribute


def train(**kwargs):
    writer = SummaryWriter(log_dir=kwargs["log_path"])
    adj_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    ]
    node_attrs = [
        Attribute.RANDOM,
        Attribute.RANDOM,
        Attribute.RANDOM,
        Attribute.RANDOM,
    ] + [Attribute.NORMAL for _ in range(8)]
    env = Env(adj_matrix, node_attrs)

    bads_n = 4
    goods_n = 8

    normals = [
        Agent(
            obs_dim=env.features_n[i + bads_n],
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
        for episode in range(episodes_n):
            acts = [None for _ in range(bads_n)] + [
                np.full(
                    (1, len(env.topology.nodes[i + bads_n].adjacents)),
                    1 / len(env.topology.nodes[i + bads_n].adjacents),
                ).squeeze()
                for i in range(goods_n)
            ]
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
            # for i, _act in enumerate(acts):
            #     if _act is None:
            #         continue
            #     writer.add_scalars(
            #         "Actions of Nodes {0}".format(i),
            #         {"Adj {0}".format(j): a.item() for adj, w in env.nodes},
            #         epoch * episodes_n + episode,
            #     )
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
