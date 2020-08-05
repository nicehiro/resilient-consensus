import time
import torch

import numpy as np
from gym.spaces import Box

from maddpg.agent import Agent
from env import Env, Property
from torch.utils.tensorboard import SummaryWriter
from utils import normalize


def train(episodes_n=int(1e7),
          epochs_n=100,
          lr=1e-4,
          actor_lr=1e-4,
          critic_lr=1e-3,
          polyak=0.95,
          noise_scale=0.1,
          restore=False,
          need_exploit=True,
          batch_size=64,
          train=True,
          memory_size=int(1e6),
          hidden_size=64,
          hidden_layer=4,
          tolerance=0.01,
          log=True,
          log_path='maddpg-logs',
          save=False,
          reset_env=True,
          evil_nodes_type='3r'):
    writer = SummaryWriter(log_dir=log_path)
    env = Env(nodes_n=10, evil_nodes_type=evil_nodes_type, reset_env=reset_env)
    X_dims = [0 for _ in range(10)]
    A_dims = [0 for _ in range(10)]
    for i, node in enumerate(env.map.nodes):
        if node.property == Property.GOOD:
            for j in node.weights.keys():
                if j != i and env.map.nodes[j].property == Property.GOOD:
                    X_dims[i] += env.features_n[j]
                    A_dims[i] += env.outputs_n[j] // 2
        else:
            continue
    agents = [Agent(node_i=i,
                    observation_space=Box(low=0, high=1, shape=[env.features_n[i], ], dtype=np.float32),
                    action_space=Box(low=0, high=1, shape=[env.outputs_n[i] // 2, ], dtype=np.float32),
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    memory_size=memory_size,
                    batch_size=batch_size,
                    noise_scale=noise_scale,
                    polyak=polyak,
                    X_dim=X_dims[i],
                    A_dim=A_dims[i],
                    evil_nodes_type=evil_nodes_type)
              if env.is_good(i) else None for i in range(10)]
    update_after = 10000
    update_every = 100
    # Prepare for interaction with environment
    total_steps = episodes_n * epochs_n
    start_steps = 10000 if train else 0
    ep_ret = [0 for _ in range(10)]
    t = 0

    # Main loop: collect experience in env and update/log each epoch
    for epi in range(episodes_n):
        o = env.reset()
        steps = 0
        ep_ret = [0 for _ in range(10)]
        for epo in range(epochs_n): 
            t += 1
            steps += 1
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            acts = []
            acts_ = []
            # [normalize(x) for x in o]
            for i, agent in enumerate(agents):
                if not agent:
                    acts.append(None)
                    acts_.append(None)
                    continue
                if t > start_steps:
                    # obs = normalize(o[i])
                    a = agent.act(o[i])
                else:
                    a = agent.action_space.sample()
                a_ = agent.ac_targ.pi(torch.as_tensor(o[i], dtype=torch.float32))
                acts.append(a)
                acts_.append(a_)

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
            d = env.is_done(tolerance)
            o_ = env.states()
            # [normalize(x) for x in o_]
            # env.update_value_of_node()

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)

            X, X_, A, A_ = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
            for i, agent in enumerate(agents):
                if not agent:
                    continue
                for j in env.map.nodes[i].weights.keys():
                    if not o[j] or j == i:
                        continue
                    X[i].append(o[j])
                    X_[i].append(o_[j])
                    A[i].append(acts[j])
                    A_[i].append(acts_[j])

            X = [x if len(x) == 0 else np.concatenate(x) for x in X]
            X_ = [x if len(x) == 0 else np.concatenate(x) for x in X_]
            A = [x if len(x) == 0 else np.concatenate(x) for x in A]
            A_ = [x if len(x) == 0 else np.concatenate(x) for x in A_]
            # Store experience to replay buffer
            for i, agent in enumerate(agents):
                if not agent:
                    continue
                agent.memory.store(o[i], acts[i], rews[i], o_[i], acts_[i], d, X[i], X_[i], A[i], A_[i])
            # Store experience to replay buffer
            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o_

            # End of trajectory handling
            if env.is_done(tolerance) or steps >= epochs_n:
                for i in range(10):
                    writer.add_scalar('Return/Node {0}'.format(i), ep_ret[i], t)
                    writer.add_scalars('Node {0} Weights'.format(i), {'Adj {0}'.format(k): v for k, v in env.map.nodes[i].weights.items()}, t)
                writer.add_scalars('Nodes', {'{0}'.format(i): env.map.nodes[i].v for i in range(10)}, t)
                print(env.map.node_val())
                break

            if not train:
                continue
            # Update handling
            if t >= update_after and t % update_every == 0:
                for i, agent in enumerate(agents):
                    if not agent:
                        continue
                    for j in range(update_every):
                        loss_q, loss_pi = agent.optimize()
                    writer.add_scalar('Loss Q/Node {0}'.format(i), loss_q, t)
                    writer.add_scalar('Loss Pi/Node {0}'.format(i), loss_pi, t)
                    agent.save()
    return env


if __name__ == '__main__':
    train()
