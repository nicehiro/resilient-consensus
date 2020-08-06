from dqn.train import train as dqn_train
from pg.train import train as pg_train
from ddpg.train import train as ddpg_train
from q_new.train import q_consensus
from rival.train import train as rival_dqn_train
from utils import batch_train
from maddpg.train import train as maddpg_train
import argparse


if __name__ == '__main__':
    # dqn_train(restore=False, need_exploit=True, train=True, hidden_layer=3, hidden_size=256,
    #           batch_size=2, memory_size=2, reset_env=False)
    # maddpg_train()
    q_consensus(reset_env=False, evil_nodes_type='3r')
    # batch_train(dqn_train, method='DQN', label='3c')
    # pg_train()
    # ddpg_train()
    # q_consensus()
#     rival_dqn_train()
