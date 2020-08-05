from dqn.train import train as dqn_train
from pg.train import train as pg_train
from ddpg.train import train as ddpg_train
from q_new.train import q_consensus
from rival.train import train as rival_dqn_train
from maddpg.train import train as maddpg_train
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
    parser.add_argument('--actor_lr', type=float, help='Actor network learning rate.')
    parser.add_argument('--critic_lr', type=float, help='Critic network learning rate.')
    parser.add_argument('--noise_scale', type=float, help='Noise scale.')
    parser.add_argument('--hidden_size', type=int, help='Layer hidden size.')
    parser.add_argument('--hidden_layer', type=int, help='Hidden layer nums.')
    parser.add_argument('--log', type=str2bool, help='Tensorboard log file.')
    parser.add_argument('--log_path', type=str, help='Tensorboard log file path.')
    parser.add_argument('--reset_env', type=str2bool, help='Reset env.')
    parser.add_argument('--batch_num', type=int, default=1, help='Batch number.')
    parser.add_argument('--save', type=str2bool, help='Save trained model.')
    parser.add_argument('--evil_nodes_type', type=str, help='Evil nodes type. 3r, 2r1c, 1r2c etc.')
    parser.add_argument('--train_method', type=str, help='Train method. DQN, DDPG .etc.')
    parser.add_argument('--tolerance', type=float, help='Done tolerance.')
    args = parser.parse_args()
    batch_num = args.batch_num
    tolerance = args.tolerance
    success_times = failed_times = 0
    for i in range(batch_num):
        print('Train Times: {0}'.format(i))
        env = eval(args.train_method)(
            episodes_n=args.episodes,
            epochs_n=args.epochs,
            restore=args.restore,
            need_exploit=args.need_exploit,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            train=args.train,
            lr=args.lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            hidden_size=args.hidden_size,
            hidden_layer=args.hidden_layer,
            log=args.log,
            log_path=args.log_path,
            reset_env=args.reset_env,
            save=args.save,
            evil_nodes_type=args.evil_nodes_type,
            tolerance=tolerance
        )
        if env.is_done(tolerance):
            success_times += 1
        else:
            failed_times += 1
        print('Success Times: {0}\tFalied Times: {1}'.format(success_times, failed_times))
    # batch_train(dqn_train, method='DQN', label='2c')
    # pg_train()
    # ddpg_train()
    # q_consensus()
#     rival_dqn_train()
