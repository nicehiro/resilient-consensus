import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from typing import List

import sys

print(sys.path)
from rcenv import Env
from attribute import Attribute
from utils import adjacent_matrix, linestyle_tuple


def draw_trajectory(path: List[str], fig_title, save_name):
    matplotlib.use("Agg")
    plt.style.use(["science", "ieee"])
    bad_node_attrs = [
        [Attribute.RANDOM] * 4,
        [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        [Attribute.CONSTANT] * 2 + [Attribute.RANDOM] * 2,
        [Attribute.CONSTANT] * 4,
    ]
    evil_nodes_types = ["rrrr", "rrcc", "ccrr", "cccc"]
    sub_fig_titles = ["RRRR", "RRCC", "CCRR", "CCCC"]
    colors = [
        "0.25",
        "0.25",
        "0.25",
        "#FFC300",
        "#2ECC71",
        "#9B59B6",
        "#E74C3C",
        "#DAF7A6",
        "#FFC300",
        "#3498DB",
        "#999999",
        "#111111",
    ]
    line_labels = ["Node{0}".format(i) for i in range(12)]
    lines = []
    if len(path) != 4:
        raise ValueError("Path should contains 4 sub path.")
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(fig_title)
    for j in range(len(path)):
        node_attrs = bad_node_attrs[j] + [Attribute.NORMAL] * 8
        env = Env(adj_matrix=adjacent_matrix, node_attrs=node_attrs)
        df = pd.read_csv(path[j])
        axs[j // 2][j % 2].set_title(sub_fig_titles[j])
        axs[j // 2][j % 2].set_xlabel("Times")
        axs[j // 2][j % 2].set_ylabel("State")
        with plt.style.context(["science", "ieee"]):
            for i in range(1, 5):
                if node_attrs[i - 1] is Attribute.RANDOM:
                    l = axs[j // 2][j % 2].plot(
                        df[df.iloc[:, 0] % 10 == 0].iloc[:, i],
                        linestyle="-",
                        linewidth=0.2,
                        color="#DCDCDC",
                    )
                if node_attrs[i - 1] is Attribute.CONSTANT:
                    l = axs[j // 2][j % 2].plot(
                        df[df.iloc[:, 0] % 100 == 0].iloc[:, i],
                        linestyle="-",
                        linewidth=0.5,
                        color="black",
                    )
                lines.append(l)
            for i in range(5, 13):
                l = axs[j // 2][j % 2].plot(
                    df[df.iloc[:, 0] % 40 == 0].iloc[:, i],
                    linestyle=linestyle_tuple[i - 5][-1],
                    linewidth=0.5,
                )
                lines.append(l)
    axs[1][1].legend(labels=line_labels, ncol=2, labelspacing=0.2)
    # plt.show()
    plt.savefig(save_name, format="eps")


if __name__ == "__main__":
    base_path = "./analysis/data/"
    # file_path = ["q_c_rrrr.csv", "q_c_rrcc.csv", "q_c_ccrr.csv", "q_c_cccc.csv"]
    file_path = [
        "q-consensus-switcher-rrrr.csv",
        "q-consensus-switcher-rrcc.csv",
        "q-consensus-switcher-ccrr.csv",
        "q-consensus-switcher-cccc.csv",
    ]
    draw_trajectory(
        list(map(lambda x: base_path + x, file_path)),
        fig_title="Q Consensus for switcher topology",
        save_name="q_consensus_switcher_trajectories.eps",
    )
