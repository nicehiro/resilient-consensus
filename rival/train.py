from dqn.agent import DQNAgent
from env import Env
from utils import writer


def train(episodes_n=100,
          epochs_n=100):
    env = Env(nodes_n=10)
    # if agent need_exploit, it means agent have to run more episode to train
    # and every episode should from start to train
    # need exploit:
    #     need_exploit = True, memory_size = int(1e4), batch_size = 16
    # one step update:
    #     need_exploit = False, memory_size = 1, batch_size = 1
    agents = [DQNAgent(node_i=i,
                       features_n=env.features_n[i],
                       actions_n=env.outputs_n[i],
                       need_exploit=False,
                       batch_size=2,
                       memory_size=2)
              if env.is_good(i) or env.is_rival(i) else None for i in range(env.nodes_n)]
    episodes_n = int(1e4)
    epochs_n = 100
    for epi in range(episodes_n):
        states = env.reset()
        for epoch in range(epochs_n):
            acts = []
            rews = []
            rival_acts = {}
            for i, agent in enumerate(agents):
                if not env.is_good(i) and not env.is_rival(i):
                    acts.append(-1)
                    rews.append(-1)
                    continue
                a = agent.act(states[i])
                r = env.step(a, i)
                acts.append(a)
                rews.append(r)
                if env.is_rival(i):
                    rival_acts[i] = a[0]
            env.update_value_of_node(rival_action_args=rival_acts)
            states_next = env.states()
            for i, agent in enumerate(agents):
                if not env.is_good(i) and not env.is_rival(i):
                    continue
                agent.memory.store(states[i], acts[i], rews[i], states_next[i])
                if (epoch % 10 == 0):
                    loss = agent.optimize_model()
            states = states_next
        for i in range(10):
            writer.add_scalars('Node {0} Weights'.format(i),
                               {'Adj {0}'.format(k): v for k, v in env.map.nodes[i].weights.items()}, epi)
        writer.add_scalars('Nodes', {'{0}'.format(i): env.map.nodes[i].v for i in range(10)}, epi)
        print(env.map.node_val())
    return env
