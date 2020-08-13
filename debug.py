from dqn.train import train as dqn_train
from pg.train import train as pg_train
from ddpg.train import train as ddpg_train
from q_new.train import q_consensus
from utils import batch_train
from maddpg.train import train as maddpg_train
from rival.maddpg_vs_maddpg.train import train as rival_maddpg_train
from rival.qnew_vs_maddpg.train import train as rival_qnew_train
import argparse


if __name__ == '__main__':
    # dqn_train(restore=False, need_exploit=True, train=True, hidden_layer=3, hidden_size=256,
    #           batch_size=2, memory_size=2, reset_env=False)
    maddpg_train(episodes_n=300,
                 epochs_n=100,
                 restore=False,
                 memory_size=int(1e4),
                 batch_size=64,
                 actor_lr=0.0001,
                 critic_lr=0.001,
                 hidden_layer=3,
                 hidden_size=512,
                 log_path='maddpg-3r-logs',
                 reset_env=True,
                 train=True,
                 save=False,
                 evil_nodes_type='3r',
                 tolerance=10,
                 save_csv=False,
                 with_noise=False)
    # q_consensus(reset_env=False, evil_nodes_type='1r2c', save_csv=True, with_noise=True)
    # batch_train(dqn_train, method='DQN', label='3c')
    # pg_train()
    # ddpg_train()
    # rival_maddpg_train(evil_nodes_type='maddpg')
    # rival_qnew_train(evil_nodes_type='maddpg', reset_env=False, actor_lr=0.00001, critic_lr=0.000001, memory_size=int(1e6), noise_scale=0.1, batch_size=1024, polyak=0.99)
