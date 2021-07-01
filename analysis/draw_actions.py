import pandas as pd
from rcenv import Env
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib
import numpy as np
import seaborn as sns

from utils import logs2df, linestyle_tuple, adjacent_matrix
from attribute import Attribute

from scipy.interpolate import make_interp_spline, BSpline
from typing import Dict, List
from collections import OrderedDict



def x_fmt(tick_val, pos):
    val = int(tick_val) / 10000
    return "{0}".format(val)


def load_actions_of_node(dfs, x=2, y=4):
    """Load actions value of each normal node from tensorboard log file,
    then draw weight training graph.

        Args:
           dfs Dict of DataFrames.
           x X dimentions. (default 2)
           y Y dimentions. (default 4)

        Returns:
            None

    """
    light_color = [
        '#E0E0E0',
        '#E0E0E0',
        '#E0E0E0',
        '#E0E0E0',
        '#FFCDD2',
        '#E1BEE7',
        '#D1C4E9',
        '#BBDEFB',
        '#B2DFDB',
        '#C8E6C9',
        '#F0F4C3',
        '#FFE0B2'
    ]
    dark_color = [
        '#424242',
        '#424242',
        '#424242',
        '#424242',
        '#B71C1C',
        '#4A148C',
        '#B388FF',
        '#1565C0',
        '#00695C',
        '#2E7D32',
        '#9E9D24',
        '#EF6C00'
    ]
    bad_attrs = [Attribute.RANDOM] * 4
    node_attrs = bad_attrs + [Attribute.NORMAL] * 8
    env = Env(adjacent_matrix, node_attrs, times=1, has_noise=True)
    matplotlib.use("Agg")
    sns.set(font='Times New Roman')

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # line_labels = [r'$\alpha_{*i}$'.format(i) for i in range(12)]
    line_labels = [r'$\alpha_{*0}$', r'$\alpha_{*1}$', r'$\alpha_{*2}$', r'$\alpha_{*3}$',
                   r'$\alpha_{*4}$', r'$\alpha_{*5}$', r'$\alpha_{*6}$', r'$\alpha_{*7}$',
                   r'$\alpha_{*8}$', r'$\alpha_{*9}$', r'$\alpha_{*10}$', r'$\alpha_{*11}$']

    fig, axs = plt.subplots(x, y, figsize=[10, 8])
    lines = {}
    for row in range(x):
        for column in range(y):
            node_i = str(row * y + column + len(bad_attrs))

            if str(node_i) not in dfs:
                continue

            j = 0
            for adj_i, adj_df in dfs[str(node_i)].items():
                i = int(adj_i)
                linestyle = (
                    "-" if env.topology.nodes[i].attribute is Attribute.NORMAL else ":"
                )
                t = adj_df[adj_df.iloc[:, 2] % 1 == 0].iloc[:, 1]
                axs[row][column].plot(
                    t,
                    color=light_color[i],
                    alpha=0.6,
                )
                l,  = axs[row][column].plot(
                    t.rolling(1000).mean(),
                    color=dark_color[i],
                    linestyle=linestyle,
                    linewidth=1.0,
                )
                if i not in lines:
                    lines[i] = l
                j += 1
            axs[row][column].xaxis.set_major_formatter(x_fmt)
            axs[row][column].set_title("Node {0}".format(node_i))

    axs[1][0].set_xlabel("Times/10000")
    axs[1][1].set_xlabel("Times/10000")
    axs[1][2].set_xlabel("Times/10000")
    axs[1][3].set_xlabel("Times/10000")
    axs[0][0].set_ylabel("Weights")
    axs[1][0].set_ylabel("Weights")
    # fig.suptitle("Training process of RL Based Algorithm")

    l = OrderedDict(sorted(lines.items()))
    fig.legend(handles=l.values(), labels=line_labels, ncol=6, loc='lower center')
    save_name = "weights_of_rl.eps"
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.95, wspace=0.35)
    # plt.show()
    plt.savefig(save_name, format="eps")


if __name__ == "__main__":
    # dfs = logs2df("./logs/rrcc-10-noise")
    dfs = logs2df("./logs/rrcc-5-big-noise")
    load_actions_of_node(dfs, 2, 4)
