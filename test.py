from dqn.train import train as dqn_train
from pg.train import train as pg_train
from ddpg.train import train as ddpg_train
from q_new.train import q_consensus
from rival.train import train as rival_dqn_train
from utils import batch_train
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', 0):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, help='Episodes times.')
    parser.add_argument('--epochs', type=int, help='Epochs times of each episode.')
    parser.add_argument('--restore', type=str2bool, help='Restore trained model.')
    parser.add_argument('--need_exploit', type=str2bool, help='Use random policy to exploit.')
    parser.add_argument('--batch_size', type=int, help='Batch size.')
    parser.add_argument('--memory_size', type=int, help='Replay buffer size.')
    parser.add_argument('--train', type=str2bool, help='Optimize model.')
    parser.add_argument('--lr', type=float, help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, help='Layer hidden size.')
    parser.add_argument('--hidden_layer', type=int, help='Hidden layer nums.')
    parser.add_argument('--train_method', type=str, help='Train method. DQN, DDPG .etc.')
    args = parser.parse_args()
    eval(args.train_method)(
        episodes_n=args.episodes,
        epochs_n=args.epochs,
        restore=args.restore,
        need_exploit=args.need_exploit,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        train=args.train,
        lr=args.lr,
        hidden_size=args.hidden_size,
        hidden_layer=args.hidden_layer
    )
    # batch_train(dqn_train, method='DQN', label='3c')
    # pg_train()
    # ddpg_train()
    # q_consensus()
#     rival_dqn_train()
