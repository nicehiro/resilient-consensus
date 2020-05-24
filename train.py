from dqn.agent import DQNAgent
from env import Env


def dqn(episodes_n=1000,
        epochs_n=20):
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
                a, o = agent.act(states[i])
                r = env.step(a, i)
                acts.append(a)
                rews.append(r)
            env.update_value_of_node()
            states_next = env.states()
            for i, agent in enumerate(agents):
                agent.memory.store(states[i], acts[i], rews[i], states_next[i])
                if (epoch % 10 == 0):
                    loss = agent.optimize_model()
                    print('Episode: {0}\tLoss: {1}'.format(epi, loss))
            states = states_next

    print(env.map.nodes.__str__())


if __name__ == '__main__':
    dqn()
