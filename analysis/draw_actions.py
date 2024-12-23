import pandas as pd

import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from rcenv import Env

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib
import numpy as np
import random
import seaborn as sns

from utils import logs2df, linestyle_tuple, adjacent_matrix
from attribute import Attribute

from scipy.interpolate import make_interp_spline, BSpline
from typing import Dict


def x_fmt(tick_val, pos):
    val = int(tick_val) / 10000
    return "{0}".format(val)


def load_actions_of_node(dfs, probs, name, noise_scale=0, x=2, y=4):
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
        "#77AADD",
        "#EE8866",
        "#EEDD88",
        "#FFAABB",
        "#99DDFF",
        "#44BB99",
        "#DDDDDD",
    ]
    dark_color = [
        "#0C5DA5",
        "#FF9500",
        "#CCBB44",
        "#FF2C00",
        "#0077BB",
        "#009988",
        "#9e9e9e",
    ]
    sns.set_theme()
    sns.set_context("paper")

    bad_attrs = [Attribute.RANDOM] * 4
    node_attrs = bad_attrs + [Attribute.NORMAL] * 8
    seeds = [random.random() * 100 for _ in range(12)]
    env = Env(adjacent_matrix, node_attrs, probs=probs, seeds=seeds, times=1, noise_scale=noise_scale)
    # matplotlib.use("Agg")
    # plt.style.use(["science"])
    fig, axs = plt.subplots(x, y, constrained_layout=True)
    for row in range(x):
        for column in range(y):
            node_i = str(row * y + column + len(bad_attrs))
            lines = []
            line_labels = []

            if str(node_i) not in dfs:
                continue

            j = 0
            for adj_i, adj_df in dfs[str(node_i)].items():
                i = int(adj_i)
                linestyle = (
                    "-" if env.topology.nodes[i].attribute is Attribute.NORMAL else ":"
                )
                marker = (
                    "" if env.topology.nodes[i].attribute is Attribute.NORMAL else "^"
                )
                # with plt.style.context(["science"]):
                t = adj_df[adj_df.iloc[:, 2] % 1 == 0].iloc[:, 1]
                axs[row][column].plot(
                    t,
                    color=light_color[j],
                    alpha=0.6,
                )
                l = axs[row][column].plot(
                    t.rolling(1000).mean(),
                    color=dark_color[j],
                    linestyle=linestyle,
                    linewidth=0.5,
                )
                lines.append(l)
                line_labels.append("Adj {0}".format(adj_i))
                j += 1
            axs[row][column].legend(
                handles=lines, labels=line_labels, loc="lower right"
            )
            axs[row][column].xaxis.set_major_formatter(x_fmt)
            axs[row][column].set_xlabel("Times/10000")
            axs[row][column].set_ylabel("Weights")
            axs[row][column].set_title("Node {0}".format(node_i))

    fig.suptitle("Training process of Modified RL")
    save_name = f"data/{name}.png"
    # plt.show()
    plt.savefig(save_name, format="png")


if __name__ == "__main__":
    # load_actions_of_node('./data/directed-2r1c-csv/')
    noise_scale = 0.0
    probs = [1.0] * 4 + [1.0] * 8

    dfs = logs2df("./logs/rrcc-1.0/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrcc-1.0-modified-rl", noise_scale, 2, 4)
    dfs = logs2df("./logs/rrcc-1.0/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrcc-1.0-ddpg", noise_scale, 2, 4)

    dfs = logs2df("./logs/rrrr-1.0/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrrr-1.0-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/rrrr-1.0/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrrr-1.0-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/cccc-1.0/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "cccc-1.0-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/cccc-1.0/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "cccc-1.0-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/rrrr-1.0-nn/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrrr-1.0-nn-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/rrrr-1.0-nn/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrrr-1.0-nn-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/cccc-1.0-nn/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "cccc-1.0-nn-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/cccc-1.0-nn/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "cccc-1.0-nn-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/rrcc-1.0-nn/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrcc-1.0-nn-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/rrcc-1.0-nn/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrcc-1.0-nn-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/rrrr-0.5/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrrr-0.5-nn-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/rrrr-0.5-nn/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrrr-0.5-nn-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/rrcc-0.5/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrcc-0.5-nn-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/rrcc-0.5/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "rrcc-0.5-nn-modified-rl", noise_scale, 2, 4)

    dfs = logs2df("./logs/cccc-0.5/ddpg", False, True, "./data")
    load_actions_of_node(dfs, probs, "cccc-0.5-nn-ddpg", noise_scale, 2, 4)
    dfs = logs2df("./logs/cccc-0.5/modified_rl", False, True, "./data")
    load_actions_of_node(dfs, probs, "cccc-0.5-nn-modified-rl", noise_scale, 2, 4)