from dqn.agent import DQNAgent
from env import Env
from utils import writer


def train(episodes_n=100,
        epochs_n=100):
    env = Env(nodes_n=10)
    agents = [DQNAgent(node_i=i,
                       features_n=env.features_n[i],
                       actions_n=env.outputs_n[i])
              for i in range(env.nodes_n)]
    for epi in range(episodes_n):
        states = env.reset()
        for epoch in range(epochs_n):
            acts = []
            rews = []
            for i, agent in enumerate(agents):
                a= agent.act(states[i])
                r = env.step(a, i)
                acts.append(a)
                rews.append(r)
            env.update_value_of_node()
            states_next = env.states()
            for i, agent in enumerate(agents):
                agent.memory.store(states[i], acts[i], rews[i], states_next[i])
                if (epoch % 10 == 0):
                    loss = agent.optimize_model()
            states = states_next
        for i in range(10):
            writer.add_scalars('Node {0} Weights'.format(i), {'Adj {0}'.format(k): v for k, v in env.map.nodes[i].weights.items()}, epi)
        writer.add_scalars('Nodes', {'{0}'.format(i): env.map.nodes[i].v for i in range(10)}, epi)
    print(env.map.__str__())
