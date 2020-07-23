from dqn.agent import DQNAgent
from env import Env
from utils import writer
import argparse


def train(episodes_n=int(1e7),
          epochs_n=100,
          lr=0.001,
          restore=False,
          need_exploit=True,
          batch_size=64,
          train=True,
          memory_size=int(1e5),
          hidden_size=64,
          hidden_layer=4,
          log=True,
          reset_env=True):
    print(train)
    env = Env(nodes_n=10, reset_env=reset_env)
    # if agent need_exploit, it means agent have to run more episode to train
    # and every episode should from start to train
    # need exploit:
    #     need_exploit = True, memory_size = int(1e4), batch_size = 16
    # one step update:
    #     need_exploit = False, memory_size = 1, batch_size = 1
    agents = [DQNAgent(node_i=i,
                       features_n=env.features_n[i],
                       actions_n=env.outputs_n[i],
                       lr=lr,
                       need_exploit=need_exploit,
                       batch_size=batch_size,
                       restore=restore,
                       train=train,
                       hidden_sizes=[hidden_size]*hidden_layer,
                       memory_size=memory_size)
              if env.is_good(i) else None for i in range(env.nodes_n)]
    success_times, failed_times, = 0, 0
    for epi in range(episodes_n):
        states = env.reset()
        rewards = [0 for _ in range(10)]
        for epoch in range(epochs_n):
            acts = []
            rews = []
            for i, agent in enumerate(agents):
                if not agent:
                    acts.append(-1)
                    rews.append(-1)
                    continue
                a = agent.act(states[i])
                r = env.step(a, i)
                acts.append(a)
                rews.append(r)
                rewards[i] += r
            env.update_value_of_node()
            states_next = env.states()
            for i, agent in enumerate(agents):
                if not agent:
                    continue
                agent.memory.store(states[i], acts[i], rews[i], states_next[i])
                if (epoch % 20 == 0):
                    loss = agent.optimize_model()
                    agent.save()
            states = states_next
        if log:
            for i in range(10):
                writer.add_scalars('Node {0} Weights'.format(i),
                                   {'Adj {0}'.format(k): v for k, v in env.map.nodes[i].weights.items()}, epi)
            writer.add_scalars('Nodes', {'{0}'.format(i): env.map.nodes[i].v for i in range(10)}, epi)
            writer.add_scalars('Rewards', {'{0}'.format(i): rewards[i] for i in range(10)}, epi)
        if not train:
            if env.is_done():
                success_times += 1
            else:
                failed_times += 1
            print('Success Times: {0}\tFailed times: {1}\tTotal Times: {2}'.format(success_times, failed_times, epi+1))
    print(env.map.node_val())
    return env
