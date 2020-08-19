import time
import torch

import numpy as np
from gym.spaces import Box

from copy import deepcopy

from rival.ddpg_vs_ddpg.agent import Agent
from env import Env, Property
from torch.utils.tensorboard import SummaryWriter


def train(**kwargs):
    writer = SummaryWriter(log_dir=kwargs['log_path'])
    env = Env(nodes_n=10, evil_nodes_type=kwargs['evil_nodes_type'], reset_env=kwargs['reset_env'])
    agents = [Agent(node_i=i,
                    observation_space=Box(low=0, high=1, shape=[env.features_n[i], ], dtype=np.float32),
                    action_space=Box(low=0, high=1, shape=[env.outputs_n[i], ], dtype=np.float32),
                    actor_lr=0.0001,
                    critic_lr=0.0001,
                    memory_size=10000,
                    batch_size=64,
                    polyak=0.9,
                    train=True,
                    hidden_layer=3,
                    hidden_size=256,
                    restore_path='./models/rival/ddpg/',
                    evil_nodes_type=kwargs['evil_nodes_type'])
                if env.is_good(i) else
             Agent(node_i=i,
                   observation_space=Box(low=0, high=1, shape=[env.features_n[i], ], dtype=np.float32),
                   action_space=Box(low=0, high=1, shape=[env.outputs_n[i], ], dtype=np.float32),
                   actor_lr=kwargs['actor_lr'],
                   critic_lr=kwargs['critic_lr'],
                   memory_size=kwargs['memory_size'],
                   batch_size=kwargs['batch_size'],
                   polyak=kwargs['polyak'],
                   train=kwargs['train'],
                   restore_path='./models/rival/ddpg/',
                   evil_nodes_type=kwargs['evil_nodes_type'])
               for i in range(10)]
    episodes_n = kwargs['episodes_n']
    epochs_n = kwargs['epochs_n']
    update_after = 10000
    update_every = 20
    start_steps = 1000
    t = 0

    # Main loop: collect experience in env and update/log each epoch
    for epi in range(episodes_n):
        o = env.reset()
        ep_ret = [0 for _ in range(10)]
        for epo in range(epochs_n): 
            t += 1
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            acts = []
            # [normalize(x) for x in o]
            for i, agent in enumerate(agents):
                if not agent:
                    acts.append(None)
                    continue
                if t > start_steps or not kwargs['train']:
                    a = agent.act(o[i])
                else:
                    a = agent.action_space.sample()
                acts.append(a)
            
            if kwargs['train']:
                for i, _act in enumerate(acts):
                    if _act is None:
                        continue
                    writer.add_scalars('Agent {0} actions'.format(i),
                                       {'{0}'.format(j): a.item() for j, a in enumerate(_act)}, t)

            rews = []
            # Step the env
            for i, agent in enumerate(agents):
                if not agent:
                    rews.append(None)
                    continue
                r = env.step(acts[i], i, is_continuous=True, update_value=False if env.is_good(i) else True)
                rews.append(r)
                ep_ret[i] += r
            env.update_value_of_node()
            d = env.is_done(0.01)
            o_ = env.states()

            # Store experience to replay buffer
            for i, agent in enumerate(agents):
                if not agent:
                    continue
                agent.memory.store(o[i], acts[i], rews[i], o_[i], d)
            # Store experience to replay buffer
            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o_

            # End of trajectory handling
            if kwargs['train']:
                for i in range(10):
                    writer.add_scalar('Return/Node {0}'.format(i), ep_ret[i], t)

            # Update handling
            if kwargs['train']:
                if t >= update_after and t % update_every == 0:
                    for i, agent in enumerate(agents):
                        if not agent:
                            continue
                        for j in range(2):
                            loss_q, loss_pi = agent.optimize()
                        writer.add_scalar('Loss Q/Node {0}'.format(i), loss_q, t)
                        writer.add_scalar('Loss Pi/Node {0}'.format(i), loss_pi, t)
                        agent.save()
    print(env.map)
    return env


if __name__ == '__main__':
    train()
