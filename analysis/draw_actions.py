import pandas as pd
from env import Env
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline


def x_fmt(tick_val, pos):
    val = int(tick_val) / 10000
    return '{0}'.format(val)


def load_actions_of_node(path: str, x=2, y=4):
    """Load actions of node_i.
    """
    matplotlib.use('Agg')
    env = Env(nodes_n=10, evil_nodes_type='2r1c', directed_graph=True)
    fig, axs = plt.subplots(x, y, constrained_layout=True, figsize=[10, 6])
    data = {}
    colors = ['#001d4a', '#33261D', '#4f345a', '#eca400', '#6e8898', '#8f2d56', '#d2a1b8', '#e4fde1', '#ab81cd', '#6b2737']
    for file_name in os.listdir(path):
        if file_name.startswith('.'):
            continue
        if os.path.isdir(os.path.join(path, file_name)):
            continue
        node, adj = file_name.split('-')
        if node not in data:
            data[node] = []
        data[node].append(file_name)
    # bright_colors = ['#3498DB', '#2ECC71', '#FFC300', '#DAF7A6', '#E74C3C']
    # dark_colors = ['#0E6655', '#1A5276', '#873600']
    for node, file_names in data.items():
        pos_x, pos_y = (int(node)-3) // y, (int(node)-3) % y
        adjs_n = env.map.nodes[int(node)].neighbors_n
        adjs_total_weight = 1 - (1 / adjs_n)
        labels = []
        alphas = []
        bad, good = 0, 0
        for k, _ in env.map.nodes[int(node)].weights.items():
            if k == int(node):
                continue
            if env.is_good(k):
                alphas.append(0.3)
                good += 1
            else:
                alphas.append(1)
                bad += 1
            labels.append('Adj {0}'.format(k))
        # colors = bright_colors[:good] + dark_colors[:bad]
        df = pd.read_csv(os.path.join(path, file_name))
        for i, file_name in enumerate(file_names):
            df2 = pd.read_csv(os.path.join(path, file_name))
            df[file_name] = df2['Value']
        df['sum'] = df.iloc[:, 2:].sum(axis=1)
        file_names.sort()
        for i, file_name in enumerate(file_names):
            axs[pos_x][pos_y].plot(df['Step'], df[file_name] / df['sum'] * adjs_total_weight, color=colors[env.map.weights_index_without_self[int(node)][i]])
        axs[pos_x][pos_y].legend(labels, loc='upper left', fontsize=8)
        axs[pos_x][pos_y].set_title('Node {0}'.format(node))
        axs[pos_x][pos_y].set_xlabel('Times / Million')
        axs[pos_x][pos_y].set_ylabel('Actions(Weights)')
        axs[pos_x][pos_y].xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    reward_path = os.path.join(path, 'rewards')
    # df = pd.read_csv(os.path.join(reward_path, '3r.csv'))
    file_names = os.listdir(reward_path)
    file_names.sort()
    for i, file_name in enumerate(file_names):
        if file_name.startswith('.'):
            continue
        df = pd.read_csv(os.path.join(reward_path, file_name))

        # xnew = np.linspace(df['Step'].min(), df['Step'].max(), 5)
        # spl = make_interp_spline(df['Step'], df['Value'], k=3)
        # smoothed = spl(xnew)
        # axs[1, 3].plot(xnew, smoothed, color=colors[i])
        axs[1, 3].plot(df['Step'], df['Value'].rolling(100).mean())

        # axs[1, 3].plot(df['Step'], df['Value'], alpha=0.5, color=colors[i])
    axs[1, 3].legend(['Node {0}'.format(i) for i in range(3, 11)], loc='upper left', fontsize=8)
    axs[1, 3].set_title('Rewards')
    axs[1, 3].set_xlabel('Times / Million')
    axs[1, 3].set_ylabel('Rewards')
    axs[1, 3].xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    fig.suptitle('D-DDPG Training Processes')
    plt.show()
    plt.savefig('ddpg-1r2c-train.eps', format='eps')


if __name__ == '__main__':
    load_actions_of_node('./data/directed-2r1c-csv/')

