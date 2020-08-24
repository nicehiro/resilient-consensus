from env import Env, Property
import math
from rival.qnew_vs_ddpg.agent import Agent
from gym.spaces import Box
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, v in node.weights.items():
            q[i][k] = init_value
    return q


def train(**kwargs):
    # print(kwargs)
    writer = SummaryWriter(log_dir=kwargs['log_path'])
    env = Env(nodes_n=10, reset_env=kwargs['reset_env'], evil_nodes_type=kwargs['evil_nodes_type'], times=1, with_noise=kwargs['with_noise'])
    Q = make_q(env, 1)
    step_size = 0.01
    start_steps = 10
    update_after = 100
    update_every = 10
    t = 0
    episodes_n = kwargs['episodes_n']
    epochs_n = kwargs['epochs_n']

    agents = [Agent(node_i=i,
                   observation_space=Box(low=0, high=1, shape=[env.features_n[i], ], dtype=np.float32),
                   action_space=Box(low=0, high=1, shape=[env.outputs_n[i], ], dtype=np.float32),
                   actor_lr=kwargs['actor_lr'],
                   critic_lr=kwargs['critic_lr'],
                   memory_size=kwargs['memory_size'],
                   batch_size=kwargs['batch_size'],
                   polyak=kwargs['polyak'],
                   train=kwargs['train'],
                   restore_path='./models/rival/qnew/',
                   evil_nodes_type=kwargs['evil_nodes_type'])
               if env.is_rival(i) else None for i in range(10)]
    if kwargs['save_csv']:
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(env.nodes_n)])
    for epi in range(episodes_n):
        t += 1
        o = env.reset()
        ep_ret = [0 for i in range(3)]
        for epo in range(epochs_n):
            t += 1
            acts = []
            for i, node in enumerate(env.map.nodes):
                if env.is_rival(i):
                    a = agents[i].action_space.sample() if t > start_steps else agents[i].act(o[i])
                    acts.append(a)
                    continue
                for j, q in Q[i].items():
                    # 0.01 3r: 0.0002     1r2c: 0.0001      2r1c: 0.0001      3c: 0.0001
                    # times=1, directed graph: 1 + 0.1 * epi, range(2000)
                    # times=1, undirected graph: 10 + 0.1 * epi, range(1000)
                    r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * (10 + 0.01 * epi))
                    Q[i][j] += step_size * (r_ij - Q[i][j])
                q_sum = sum(Q[i].values())
                for j in Q[i].keys():
                    w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i])) + 0.001
                    env.map.update_by_weight(i, j, w)
                env.map.normalize(i)
            rews = []
            for i, agent in enumerate(agents):
                if not agent:
                    rews.append(None)
                    continue
                r = env.step(acts[i], i, is_continuous=True, update_value=True)
                rews.append(r)
                ep_ret[i] += r
            env.map.update_value_of_node(with_noise=kwargs['with_noise'])
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
            # log
            for i, node in enumerate(env.map.nodes):
                writer.add_scalars('Agent {0} actions'.format(i), {'{0}'.format(j): v for j, v in node.weights.items()}, t)
        
        # End of trajectory handling
        if kwargs['train']:
            for i in range(10):
                if agents[i] is not None:
                    writer.add_scalar('Return/Node {0}'.format(i), ep_ret[i], t)
        print(env.map)
    with_noise_str = '' if not kwargs['with_noise'] else 'noise_'
    directed_str = '' if not kwargs['directed_graph'] else 'directed_'
    return env