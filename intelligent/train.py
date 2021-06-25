import numpy as np
from gym.spaces import Box

from intelligent.agent import Agent
from rcenv import Env
from attribute import Attribute
from utils import adjacent_matrix
from torch.utils.tensorboard import SummaryWriter


def train(**kwargs):
    writer = SummaryWriter(kwargs["log_path"])
    node_attrs = [Attribute.INTELLIGENT] * 4 + [Attribute.NORMAL] * 8
    env = Env(
        adjacent_matrix,
        node_attrs=node_attrs,
        probs=kwargs["probs"],
        seeds=kwargs["seeds"],
        times=1,
        noise_scale=kwargs["noise_scale"],
    )

    bads_n = 4
    goods_n = 8

    intelligents = [
        Agent(
            node_i=i,
            observation_space=Box(
                low=0,
                high=1,
                shape=[
                    env.features_n[i],
                ],
                dtype=np.float32,
            ),
            action_space=Box(
                low=0,
                high=1,
                shape=[
                    env.actions_n[i],
                ],
                dtype=np.float32,
            ),
        )
        for i in range(bads_n)
    ]

    episodes_n = kwargs["episodes_n"]
    epochs_n = kwargs["epochs_n"]
    update_after = 10000
    update_every = 10
    start_steps = 1000
    t = 0

    # Main loop: collect experience in env and update/log each epoch
    for epi in range(episodes_n):
        o = env.reset()
        ep_ret = [0] * bads_n
        for epo in range(epochs_n):
            t += 1
            # act
            acts = []
            for i, agent in enumerate(intelligents):
                if t > start_steps:
                    a = agent.act(o[i])
                else:
                    a = agent.action_space.sample()
                acts.append(a)

            # update value of node
            for i, node in enumerate(env.topology.nodes):
                value = acts[i] if i < len(acts) else None
                node.update_value(value=value)
            # get reward
            rews = env._reward()
            # get next observation
            o_ = env._state()
            d = False

            # Store experience to replay buffer
            for i, intelligent in enumerate(intelligents):
                intelligent.memory.store(o[i], acts[i], rews[i], o_[i], d)

            # reset observation
            o = o_

            # log normal node value
            writer.add_scalars(
                "Value of Normal Node",
                {
                    "Node {0}".format(i): env.topology.nodes[i].value
                    for i in range(bads_n, env.n)
                },
                t,
            )
            # log intelligent node value
            writer.add_scalars(
                "Value of Intelligent Node",
                {
                    "Node {0}".format(i): env.topology.nodes[i].value
                    for i in range(bads_n)
                },
                t,
            )
            # log loss
            if t > update_after and t % update_every == 0:
                for i, intelligent in enumerate(intelligents):
                    loss_q, loss_p = intelligent.optimize()
                    writer.add_scalar("Loss Q/Node {0}".format(i), loss_q, t)
                    writer.add_scalar("Loss Pi/Node {0}".format(i), loss_p, t)
        writer.add_scalar("Distance of Env", env.topology.hard_distance(), t)


if __name__ == "__main__":
    probs = [1.0] * 4 + [1.0] * 8
    train(
        probs=probs,
        noise_scale=0.01,
        seeds=[i for i in range(12)],
        save_csv=True,
        episodes_n=3000,
        epochs_n=50,
        log_path="./logs/intelligent/",
    )
