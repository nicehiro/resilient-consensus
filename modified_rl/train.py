from env import Env
from torch.utils.tensorboard import SummaryWriter
from modified_rl.agent import Agent
import numpy as np


def train(**kwargs):
    writer = SummaryWriter(log_dir=kwargs["log_path"])
    env = Env(
        nodes_n=10,
        evil_nodes_type=kwargs["evil_nodes_type"],
        reset_env=False,
        directed_graph=kwargs["directed_graph"],
    )
    bads = []
    bads_n = env.nodes_n - env.goods_n
    normals = [
        Agent(
            obs_dim=env.features_n[i + bads_n],
            act_dim=env.outputs_n[i + bads_n] + 1,
            buffer_size=kwargs["memory_size"],
            lr=kwargs["actor_lr"],
            gamma=0.95,
            evil_nodes_type=kwargs["evil_nodes_type"],
            node_i=i + bads_n,
            restore_path=kwargs["restore_path"],
            batch_size=kwargs["batch_size"],
            value=env.map.nodes[i + bads_n].v,
            node_index_of_weights=env.map.node_index_of_weights[i + bads_n],
        )
        for i in range(env.goods_n)
    ]

    epochs_n = kwargs["epochs_n"]
    episodes_n = kwargs["episodes_n"]
    update_after = kwargs["update_after"]
    update_every = kwargs["update_every"]

    for epoch in range(epochs_n):
        # sample init value of x
        o = env.reset()
        for episode in range(episodes_n):
            # save to replay buffer
            acts = [
                np.full(
                    (1, len(env.map.nodes[i + bads_n].weights)),
                    1 / len(env.map.nodes[i + bads_n].weights),
                ).squeeze()
                for i in range(env.goods_n)
            ]
            rews = []
            for i, normal in enumerate(normals):
                a = normal.act(o[i + bads_n], acts[i])
                r = env.step(a, i + bads_n, is_continuous=False)
                acts[i] = a
                rews.append(r)
            # update value
            env.update_value_of_node()
            for i, normal in enumerate(normals):
                normal.update_value(env.map.nodes[i].v)
            d = False
            o_ = env.states()
            for i, normal in enumerate(normals):
                normal.memory.store(o[i + bads_n], acts[i], rews[i], o_[i + bads_n], d)
            o = o_
            # record a to tensorboard
            for i, _act in enumerate(acts):
                writer.add_scalars(
                    "Actions of Nodes {0}".format(i),
                    {"Adj {0}".format(j): a.item() for j, a in enumerate(_act)},
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
