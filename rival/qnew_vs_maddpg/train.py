from env import Env, Property
import math
from rival.qnew_vs_maddpg.agent import Agent
from gym.spaces import Box
import numpy as np


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, v in node.weights.items():
            q[i][k] = init_value
    return q


def train(**kwargs):
    print(kwargs)
    env = Env(nodes_n=10, reset_env=kwargs['reset_env'], evil_nodes_type=kwargs['evil_nodes_type'], times=1000)
    Q = make_q(env, 1)
    step_size = 0.1
    o = env.reset()

    X_dims = [0 for _ in range(10)]
    A_dims = [0 for _ in range(10)]
    for i, node in enumerate(env.map.nodes):
        if node.property == Property.GOOD:
            for j in node.weights.keys():
                if j != i and env.map.nodes[j].property == Property.GOOD:
                    X_dims[i] += env.features_n[j]
                    A_dims[i] += env.outputs_n[j] // 2
        elif node.property == Property.MADDPG:
            for j, node_ in enumerate(env.map.nodes):
                if j != i and env.map.nodes[j].property == Property.MADDPG:
                    X_dims[i] += env.features_n[j]
                    A_dims[i] += env.outputs_n[j]
        else:
            continue
    evil_agents = [Agent(node_i=i,
                         observation_space=Box(low=0, high=1, shape=[env.features_n[i], ], dtype=np.float32),
                         action_space=Box(low=0, high=1, shape=[env.outputs_n[i], ], dtype=np.float32),
                         actor_lr=kwargs['actor_lr'],
                         critic_lr=kwargs['critic_lr'],
                         memory_size=kwargs['memory_size'],
                         batch_size=kwargs['batch_size'],
                         noise_scale=kwargs['noise_scale'],
                         polyak=kwargs['polyak'],
                         X_dim=X_dims[i],
                         A_dim=A_dims[i],
                         evil_nodes_type=kwargs['evil_nodes_type'],
                         train=False,
                         property=Property.MADDPG)
                    if not env.is_good(i) else None
                    for i in range(10)]
    for epi in range(1000):
        for i, node in enumerate(env.map.nodes):
            if node.property is Property.GOOD:
                for j, q in Q[i].items():
                    r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * 0.01)
                    Q[i][j] += step_size * (r_ij - Q[i][j])
                q_sum = sum(Q[i].values())
                for j in Q[i].keys():
                    w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i])) + 0.001
                    env.map.update_by_weight(i, j, w)
                env.map.normalize(i)
            elif node.property is Property.MADDPG:
                a = evil_agents[i].act(o[i])
                r = env.step(a, i, is_continuous=True, update_value=True)
        env.map.update_value_of_node()
        o_ = env.states()
    return env
