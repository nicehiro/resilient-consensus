import torch
from env import Env
from pg.agent import Agent
from gym.spaces import Box, Discrete
import numpy as np
from utils import writer, normalize


def train():
    epochs = 2000
    local_steps_per_epoch = 1
    env = Env(nodes_n=10, times=1)
    agents = [Agent(node_i=i,
                    observation_space=Box(low=0, high=1, shape=[env.features_n[i],], dtype=np.float32),
                    action_sapce=Box(low=0, high=1, shape=[env.outputs_n[i]//2,], dtype=np.float32))
              for i in range(10)]
    for epoch in range(epochs):
        s = env.reset()
        for t in range(local_steps_per_epoch):
            acts = []
            vs = []
            logps = []
            rews = []
            for i, agent in enumerate(agents):
                a, v, logp = agent.act(s[i])
                acts.append(a)
                vs.append(v)
                logps.append(logp)
                r = env.step(a, i, is_continuous=True)
                writer.add_scalar('Rewards/Node: {0}'.format(i), r, epoch * local_steps_per_epoch + t)
                rews.append(r)

            # save
            for i, agent in enumerate(agents):
                s[i] = normalize(s[i])
                agent.memory.store(s[i], acts[i], rews[i], vs[i], logps[i])
                
            env.update_value_of_node()
            s_ = env.states()

            # Update obs (critical!)
            s = s_

        for i, agent in enumerate(agents):
            s[i] = normalize(s[i])
            _, v, _ = agent.ac.step(torch.as_tensor(s[i], dtype=torch.float32))
            agent.memory.finish_path(v)
#             if t % 10 == 0:
            actor_loss, critic_loss = agent.optimize()
            if actor_loss != 0 and critic_loss != 0:
                writer.add_scalar('Loss_Actor/Node: {0}'.format(i), actor_loss, epoch * local_steps_per_epoch + t)
                writer.add_scalar('Loss_Critic/Node: {0}'.format(i), critic_loss, epoch * local_steps_per_epoch + t)
        for i, node in enumerate(env.map.nodes):
            for k, v in node.weights.items():
                writer.add_scalar('Node {0}/Weight {1}'.format(i, k), v, epoch)
        writer.add_scalars('Node Values', {
            'Node {0}'.format(i): env.map.nodes[i].v 
            for i in range(env.nodes_n)
        }, epoch)
#         if epoch % 100 == 0:
#             print('Episode {0}'.format(epoch))
#             print(env.map)
        if env.is_done(0.05):
            break
    writer.close()
    print('Train finished!')
    print('Episode {0}'.format(epoch))
    print(env.map)
    return env.map


if __name__ == '__main__':
    train()