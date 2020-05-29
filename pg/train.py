import torch
from env import Env
from pg.agent import Agent
from gym.spaces import Box, Discrete
import numpy as np
from utils import writer


def train():
    epochs = 10000
    local_steps_per_epoch = 100
    env = Env(nodes_n=10)
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
                rews.append(r)

            # save
            for i, agent in enumerate(agents):
                agent.memory.store(s[i], acts[i], rews[i], vs[i], logps[i])
                
            env.update_value_of_node()
            s_ = env.states()

            # Update obs (critical!)
            s = s_

        for i, agent in enumerate(agents):
            _, v, _ = agent.ac.step(torch.as_tensor(s[i], dtype=torch.float32))
            agent.memory.finish_path(v)
#             if t % 10 == 0:
            actor_loss, critic_loss = agent.optimize()
            if actor_loss != 0 and critic_loss != 0:
                writer.add_scalar('Loss/Actor', actor_loss, epoch * local_steps_per_epoch + t)
                writer.add_scalar('Loss/Critic', critic_loss, epoch * local_steps_per_epoch + t)
        writer.add_scalars('Node Values', {
            'Node {0}'.format(i): env.map.nodes[i].v 
            for i in range(env.nodes_n)
        }, epoch)
        if epoch % 100 == 0:
            print('Episode {0}'.format(epoch))
            print(env.map)
    writer.close()
    print('Train finished!')
    print('Episode {0}'.format(epoch))
    print(env.map)


if __name__ == '__main__':
    train()