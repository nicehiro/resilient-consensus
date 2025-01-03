import time

import numpy as np
import pandas as pd
from gym.spaces import Box

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from ddpg.agent import Agent
from env import Env
from torch.utils.tensorboard import SummaryWriter
from utils import normalize


def train(**kwargs):
    env = Env(
        nodes_n=12,
        evil_nodes_type=kwargs["evil_nodes_type"],
        reset_env=kwargs["reset_env"],
        directed_graph=kwargs["directed_graph"],
        with_noise=kwargs["with_noise"],
        noise_scale=kwargs["noise_scale"],
    )
    writer = SummaryWriter(kwargs["log_path"])
    agents = [
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
                    env.outputs_n[i],
                ],
                dtype=np.float32,
            ),
            train=kwargs["train"],
            evil_nodes_type=kwargs["evil_nodes_type"],
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
        )
        if env.is_good(i)
        else None
        for i in range(10)
    ]
    episodes_n = kwargs["episodes_n"]
    epochs_n = kwargs["epochs_n"]
    update_after = kwargs["update_after"]
    update_every = kwargs["update_every"]
    start_steps = 1000
    t = 0

    if kwargs["save_csv"]:
        df = pd.DataFrame(columns=["Node{0}".format(i) for i in range(env.nodes_n)])

    os.makedirs(kwargs["restore_path"], exist_ok=True)

    # Main loop: collect experience in env and update/log each epoch
    for epi in range(episodes_n):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        o, ep_len = env.reset(), 0
        ep_ret = [0 for _ in range(10)]
        for epo in range(epochs_n):
            if kwargs["save_csv"]:
                df = df.append(env.map.node_val(), ignore_index=True)
            t += 1
            acts = []
            # [normalize(x) for x in o]
            for i, agent in enumerate(agents):
                if not agent:
                    acts.append(None)
                    continue
                if t > start_steps or not kwargs["train"]:
                    # obs = normalize(o[i])
                    a = agent.act(o[i])
                else:
                    a = agent.action_space.sample()
                acts.append(a)

            if kwargs["train"]:
                for i, _act in enumerate(acts):
                    if _act is None:
                        continue
                    writer.add_scalars(
                        "Actions of Node {0}".format(i),
                        {"Adj {0}".format(j): a.item() for j, a in enumerate(_act)},
                        t,
                    )

            rews = []
            # Step the env
            for i, agent in enumerate(agents):
                if not agent:
                    rews.append(None)
                    continue
                r = env.step(acts[i], i, is_continuous=True)
                rews.append(r)
                ep_ret[i] += r
            env.update_value_of_node()
            d = env.is_done(0.01)
            o_ = env.states()
            # [normalize(x) for x in o_]
            # env.update_value_of_node()
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False

            # Store experience to replay buffer
            for i, agent in enumerate(agents):
                if not agent:
                    continue
                agent.memory.store(o[i], acts[i], rews[i], o_[i], d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o_

            if kwargs["train"]:
                # Update handling
                if t % update_every == 0:
                    for i, agent in enumerate(agents):
                        if not agent:
                            continue
                        # for j in range(2):
                        loss_q, loss_pi = agent.optimize()
                        writer.add_scalar("Loss Q of Node {0}".format(i), loss_q, t)
                        writer.add_scalar("Loss Pi of Node {0}".format(i), loss_pi, t)

        # End of trajectory handling
        if kwargs["train"]:
            for i in range(10):
                writer.add_scalar("Return/Node {0}".format(i), ep_ret[i], t)
    with_noise_str = "" if not kwargs["with_noise"] else "noise_"
    if kwargs["save_csv"]:
        df.to_csv("ddpg_{0}{1}.csv".format(with_noise_str, kwargs["evil_nodes_type"]))
    print(env.map)
    return env


if __name__ == "__main__":
    train(
        evil_nodes_type="4c",
        reset_env=True,
        directed_graph=True,
        log_path="logs/ddpg/",
        train=True,
        memory_size=1000,
        batch_size=640,
        polyak=0.95,
        hidden_layer=1,
        hidden_size=256,
        actor_lr=1e-3,
        critic_lr=1e-3,
        save_csv=False,
        with_noise=True,
        noise_scale=0.05,
        episodes_n=4000,
        epochs_n=50,
        update_after=10,
        update_every=10,
        restore_path="logs/ddpg/model/",
    )
