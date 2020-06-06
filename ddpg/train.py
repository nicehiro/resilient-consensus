import time

import numpy as np
from gym.spaces import Box

from ddpg.agent import Agent
from env import Env
from torch.utils.tensorboard import SummaryWriter


def train():
    env = Env(nodes_n=10)
    writer = SummaryWriter('logs')
    agents = [Agent(node_i=i,
                    observation_space=Box(low=0, high=1, shape=[env.features_n[i], ], dtype=np.float32),
                    action_space=Box(low=0, high=1, shape=[env.outputs_n[i] // 2, ], dtype=np.float32))
              for i in range(10)]
    steps_per_epoch = 10
    max_ep_len = 10
    update_after = 500
    update_every = 3
    epochs = 1000
    batch_size = 16
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_steps = 100
    o, ep_len = env.reset(), 0
    ep_ret = [0 for _ in range(10)]

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        acts = []
        for i, agent in enumerate(agents):
            if t > start_steps:
                a = agent.act(o[i])
            else:
                a = agent.action_space.sample()
            acts.append(a)

        rews = []
        # Step the env
        for i, agent in enumerate(agents):
            r = env.step(acts[i], i, is_continuous=True)
            rews.append(r)
            ep_ret[i] += r
        env.update_value_of_node()
        d = env.is_done(0.01)
        o_ = env.states()
        env.update_value_of_node()
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        for i, agent in enumerate(agents):
            agent.memory.store(o[i], acts[i], rews[i], o_[i], d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o_

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            for i in range(10):
                writer.add_scalar('Return/Node {0}'.format(i), ep_ret[i], t)
            # ac.save()
            o, ep_len = o_, 0
            ep_ret = [0 for _ in range(10)]

        # Update handling
        if t >= update_after and t % update_every == 0:
            for i, agent in enumerate(agents):
                for j in range(update_every):
                    loss_q, loss_pi = agent.optimize()
                writer.add_scalar('Loss Q/Node {0}'.format(i), loss_q, t)
                writer.add_scalar('Loss Pi/Node {0}'.format(i), loss_pi, t)


if __name__ == '__main__':
    train()