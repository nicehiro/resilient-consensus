import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


from rcenv import Env
from torch.utils.tensorboard import SummaryWriter
from ddpg.agent import Agent
import numpy as np
from attribute import Attribute
from utils import adjacent_matrix
import random
from gym.spaces import Box


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
            node_i=i + bads_n,
            observation_space=Box(
                low=0,
                high=1,
                shape=[
                    env.features_n[i+ bads_n]-1,
                ],
                dtype=np.float32,
            ),
            action_space=Box(
                low=0,
                high=1,
                shape=[
                    env.actions_n[i + bads_n],
                ],
                dtype=np.float32,
            ),
            train=True,
            memory_size=kwargs["memory_size"],
            batch_size=kwargs["batch_size"],
            polyak=kwargs["polyak"],
            gamma=0.95,
            hidden_layer=kwargs["hidden_layer"],
            hidden_size=kwargs["hidden_size"],
            actor_lr=kwargs["actor_lr"],
            critic_lr=kwargs["critic_lr"],
            noise_scale=0.0,
            restore_path=kwargs["restore_path"],
            value=env.topology.nodes[i + bads_n].value,
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
                a = normal.act(o[i + bads_n])
                acts[i + bads_n] = a
            rews, o_ = env.step(acts)
            # update value
            for i, normal in enumerate(normals):
                normal.update_value(env.topology.nodes[i + bads_n].value)
            d = False
            # save to replay buffer
            for i, normal in enumerate(normals):
                normal.memory.store(
                    o[i + bads_n], acts[i + bads_n], rews[i + bads_n], o_[i + bads_n], d
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
            loss_q = []
            loss_pi = []
            for normal in normals:
                l_q, l_pi = normal.optimize()
                loss_q.append(l_q)
                loss_pi.append(l_pi)
                # normal.memory.reset()
            # record loss to tensorboard
            writer.add_scalars(
                "Loss Q of Nodes",
                {"Node {0}".format(i + bads_n): l for i, l in enumerate(loss_q)},
                epoch,
            )
            writer.add_scalars(
                "Loss Pi of Nodes",
                {"Node {0}".format(i + bads_n): l for i, l in enumerate(loss_pi)},
                epoch,
            )
        if epoch % 10 == 0:
            # make dir
            os.makedirs(kwargs["restore_path"], exist_ok=True)
            for agent in normals:
                agent.save()


if __name__ == "__main__":
    probs = [0.5] * 4 + [0.5] * 8
    train(
        bad_attrs="cccc",
        probs=probs,
        noise_scale=0.01,
        log_path="logs/cccc-0.5/ddpg/",
        memory_size=1000,
        polyak=0.99,
        actor_lr=1e-4,
        critic_lr=1e-4,
        restore_path="logs/cccc-0.5/ddpg/model/",
        batch_size=64,
        epochs_n=40000,
        episodes_n=50,
        update_after=10,
        update_every=10,
        hidden_layer=1,
        hidden_size=256,
    )
