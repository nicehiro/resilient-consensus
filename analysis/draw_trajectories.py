import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from typing import List

import sys

print(sys.path)
from rcenv import Env
from attribute import Attribute
from utils import adjacent_matrix, linestyle_tuple
import seaborn as sns


def draw_trajectory(path: List[str], fig_title, save_name, save, sub_adjust, ylabels):
    if save:
        matplotlib.use("Agg")

    sns.set(font='Times New Roman')

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    l = len(path)
    rows = l // 3
    bad_node_attrs = [
        [Attribute.RANDOM] * 4,
        [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        [Attribute.CONSTANT] * 4,
    ] * rows
    sub_fig_titles = ["RRRR", "RRCC", "CCCC"] * 4
    colors = [
        "#FFC300",
        "#2ECC71",
        "#9B59B6",
        "#E74C3C",
        "#DAF7A6",
        "#FFC300",
        "#3498DB",
        "#81b214",
    ]
    line_labels = ["Node{0}".format(i) for i in range(12)]
    lines = []
    axs_x, axs_y = rows, 3
    fig, axs = plt.subplots(axs_x, axs_y, sharey=True, figsize=[10, 8])
    fig.tight_layout()
    fig.subplots_adjust(left=sub_adjust[0], right=sub_adjust[1], top=sub_adjust[2], bottom=sub_adjust[3], wspace=sub_adjust[4], hspace=sub_adjust[5])
    # fig.subplots_adjust(left=0.09, bottom=0.15, right=0.98, top=0.9, wspace=0.1, hspace=0.28)
    # fig.suptitle(fig_title)
    for j in range(len(path)):
        node_attrs = bad_node_attrs[j] + [Attribute.NORMAL] * 8
        env = Env(adj_matrix=adjacent_matrix, node_attrs=node_attrs)
        df = pd.read_csv(path[j])
        for i in range(1, 5):
            if node_attrs[i - 1] is Attribute.RANDOM:
                l = axs[j // axs_y][j % axs_y].plot(
                    df[df.iloc[:, 0] % 1 == 0].iloc[:, i],
                    linestyle=":",
                    linewidth=0.8,
                    color="#bdc7c9",
                )
            if node_attrs[i - 1] is Attribute.CONSTANT:
                l = axs[j // axs_y][j % axs_y].plot(
                    df[df.iloc[:, 0] % 1 == 0].iloc[:, i],
                    linestyle=":",
                    linewidth=0.8,
                    color="#bdc7c9",
                )
            lines.append(l)
        for i in range(5, 13):
            l = axs[j // axs_y][j % axs_y].plot(
                df[df.iloc[:, 0] % 1 == 0].iloc[:, i],
                # linestyle=linestyle_t`uple[i - 5][-1],
                linestyle="-",
                linewidth=1.0,
                color=colors[i-5]
            )
            lines.append(l)

    # axs[0][0].set(ylabel='90% Connection Probability\nState')
    # axs[1][0].set(ylabel='50% Connection Probability\nState')
    # axs[2][0].set(ylabel='10% Connection Probability\nState')
    # axs[0][0].set(ylabel='100% to be Faulty\nState')
    # axs[1][0].set(ylabel='80% to be Faulty\nState')
    # axs[2][0].set(ylabel='50% to be Faulty\nState')
    # axs[3][0].set(ylabel='10% to be Faulty\nState')

    # axs[0][0].set(ylabel='100% to be Faulty\nState')
    # axs[1][0].set(ylabel='50% to be Faulty\nState')
    # axs[2][0].set(ylabel='10% to be Faulty\nState')
    for r in range(rows):
        axs[r][0].set(ylabel=ylabels[r])
    for ax in axs.flat:
        ax.set(xlabel='Times')
        ax.set_ylim([0, 1])
        # hide nolastrow, nofirstcol label
        lastrow = ax.is_last_row()
        firstcol = ax.is_first_col()
        if not lastrow:
            # for label in ax.get_xticklabels(which="both"):
                # label.set_visible(False)
            ax.get_xaxis().get_offset_text().set_visible(False)
            ax.set_xlabel("")
        if not firstcol:
            for label in ax.get_yticklabels(which="both"):
                label.set_visible(False)
            ax.get_yaxis().get_offset_text().set_visible(False)
            ax.set_ylabel("")
    axs[0][0].set_title('RRRR')
    axs[0][1].set_title('RRCC')
    axs[0][2].set_title('CCCC')

    fig.legend(labels=line_labels, ncol=6, loc='lower center')
    if save:
        plt.savefig(save_name, format="eps")
    else:
        plt.show()


def draw_one_trajectory(file_path, save, save_name, title):
    if save:
        matplotlib.use("Agg")


    sns.set(font='Times New Roman')

    SMALL_SIZE = 14
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    bad_node_attr = [Attribute.CONSTANT] * 4
    colors = [
        "#FFC300",
        "#2ECC71",
        "#9B59B6",
        "#E74C3C",
        "#DAF7A6",
        "#FFC300",
        "#3498DB",
        "#81b214",
    ]
    line_labels = ["Node{0}".format(i) for i in range(12)]
    lines = []

    node_attrs = bad_node_attr + [Attribute.NORMAL] * 8
    df = pd.read_csv(file_path)
    for i in range(1, 5):
        if node_attrs[i - 1] is Attribute.RANDOM:
            l = plt.plot(
                df[df.iloc[:, 0] % 1 == 0].iloc[:, i],
                linestyle=":",
                linewidth=0.8,
                color="#bdc7c9",
            )
        if node_attrs[i - 1] is Attribute.CONSTANT:
            l = plt.plot(
                df[df.iloc[:, 0] % 1 == 0].iloc[:, i],
                linestyle=":",
                linewidth=0.8,
                color="#bdc7c9",
            )
        lines.append(l)
    for i in range(5, 13):
        l = plt.plot(
            df[df.iloc[:, 0] % 1 == 0].iloc[:, i],
            # linestyle=linestyle_t`uple[i - 5][-1],
            linestyle="-",
            linewidth=1.0,
            color=colors[i-5]
        )
        lines.append(l)

    plt.legend(labels=line_labels, ncol=3)
    plt.subplots_adjust()
    # plt.title(title)
    plt.xlabel('Times')
    plt.ylabel('50% to be Faulty\nState')

    plt.subplots_adjust(left=0.17, right=0.945, top=0.945, bottom=0.149)
    if save:
        plt.savefig(save_name, format="eps")
    else:
        plt.show()


if __name__ == "__main__":
    base_path = "./analysis/data/q-c/"
    fixed_path = 'fixed-2/'
    switching_path = 'switcher-2/'
    rl_path = 'rl/'
    rl_noise_path = 'rl-noise/'

    fixed_noise_path = 'fixed-noise/'

    switching_noise_path = 'switch-noise/'

    rl_noise_file_path_2 = ['rl_rrrr_10_w.csv', 'rl_rrcc_10_w.csv', 'rl_cccc_10_w.csv'] +\
                           ['rl_rrrr_10.csv', 'rl_rrcc_10.csv', 'rl_cccc_10.csv']

    rl_noise_file_path = ['rl_rrrr_10.csv', 'rl_rrcc_10.csv', 'rl_cccc_10.csv'] + \
                         ['rl_rrrr_5.csv', 'rl_rrcc_5.csv', 'rl_cccc_5.csv'] + \
                         ['rl_rrrr_1.csv', 'rl_rrcc_1.csv', 'rl_cccc_1.csv']
    rl_file_path = ["rl_rrrr_10.csv", "rl_rrcc_10.csv", "rl_cccc_10.csv"] + \
                   ["rl_rrrr_5.csv", "rl_rrcc_5.csv", "rl_cccc_5.csv"] + \
                ["rl_rrrr_1.csv", "rl_rrcc_1.csv", "rl_cccc_1.csv"]
    switching_file_path = ["q_c_switcher_rrrr_9.csv", "q_c_switcher_rrcc_9.csv", "q_c_switcher_cccc_9.csv"] + \
                ["q_c_switcher_rrrr_5.csv", "q_c_switcher_rrcc_5.csv", "q_c_switcher_cccc_5.csv"] + \
                ["q_c_switcher_rrrr_1.csv", "q_c_switcher_rrcc_1.csv", "q_c_switcher_cccc_1.csv"]
    fixed_file_path = ['q_c_rrrr_10.csv', 'q_c_rrcc_10.csv', 'q_c_cccc_10.csv'] + \
                ['q_c_rrrr_8.csv', 'q_c_rrcc_8.csv', 'q_c_cccc_8.csv'] + \
                ['q_c_rrrr_5.csv', 'q_c_rrcc_5.csv', 'q_c_cccc_5.csv'] + \
                ['q_c_rrrr_1.csv', 'q_c_rrcc_1.csv', 'q_c_cccc_1.csv']

    # rl based
    # draw_trajectory(
    #     list(map(lambda x: base_path + rl_path + x, rl_file_path)),
    #     fig_title="Distributed RL Based Method for Fixed Topology",
    #     save_name="./analysis/data/q-c/rl_fixed.eps",
    #     save=False,
    #     sub_adjust=[0.08, 0.98, 0.88, 0.18, 0.1, 0.175],
    #     ylabels=['100% to be Faulty\nState', '50% to be Faulty\nState', '10% to be Faulty\nState'],
    # )

    # q-c fixed
    # draw_trajectory(
    #     list(map(lambda x: base_path + fixed_path + x, fixed_file_path)),
    #     fig_title="Q-Consensus for Fixed Topology",
    #     save_name="./analysis/data/q-c/q_c_fixed.eps",
    #     save=False,
    #     sub_adjust=[0.09, 0.98, 0.9, 0.15, 0.1, 0.28],
    #     ylabels=['100% to be Faulty\nState', '80% to be Faulty\nState', '50% to be Faulty\nState', '10% to be Faulty\nState'],
    # )

    # q-c switching
    # draw_trajectory(
    #     list(map(lambda x: base_path + switching_path + x, switching_file_path)),
    #     fig_title="Q-Consensus for Switching Topology",
    #     save_name="./analysis/data/q-c/q_c_switching.eps",
    #     save=False,
    #     sub_adjust=[0.08, 0.98, 0.88, 0.18, 0.1, 0.175],
    #     ylabels=['90% Connection Probability\nState', '50% Connection Probability\nState', '10% Connection Probability\nState'],
    # )

    # rl based noise
    # draw_trajectory(
    #     list(map(lambda x: base_path + rl_noise_path + x, rl_noise_file_path)),
    #     fig_title="RL based algorithm for Fixed Topology",
    #     save_name="./analysis/data/q-c/rl_noise.eps",
    #     save=True,
    #     sub_adjust=[0.08, 0.98, 0.89, 0.14, 0.1, 0.175],
    #     ylabels=['100% to be Faulty\nState', '50% to be Faulty\nState', '10% to be Faulty\nState'],
    # )

    # draw_trajectory(
    #     list(map(lambda x: base_path + rl_noise_path + x, rl_noise_file_path_2)),
    #     fig_title="RL based algorithm for Fixed Topology",
    #     save_name="./analysis/data/q-c/rl_noise_2.eps",
    #     save=True,
    #     sub_adjust=[0.08, 0.98, 0.88, 0.18, 0.1, 0.175],
    #     ylabels=['Keep original weights\nState', 'Make low weights to zero\nState'],
    # )


    # q-c fixed noise
    # draw_trajectory(
    #     list(map(lambda x: base_path + fixed_noise_path + x, fixed_file_path)),
    #     fig_title="Q-Consensus for Fixed Topology",
    #     save_name="./analysis/data/q-c/q_c_fixed_noise.eps",
    #     save=True,
    #     sub_adjust=[0.1, 0.98, 0.95, 0.175, 0.175, 0.28],
    #     ylabels=['100% to be Faulty\nState', '80% to be Faulty\nState', '50% to be Faulty\nState', '10% to be Faulty\nState'],
    # )

    # q-c switching noise
    # draw_trajectory(
    #     list(map(lambda x: base_path + switching_noise_path + x, switching_file_path)),
    #     fig_title="Q-Consensus for Switching Topology",
    #     save_name="./analysis/data/q-c/q_c_switching_noise.eps",
    #     save=True,
    #     sub_adjust=[0.1, 0.98, 0.95, 0.175, 0.1, 0.34],
    #     ylabels=['90% Connection Probability\nState', '50% Connection Probability\nState', '10% Connection Probability\nState'],
    # )

    # noise_scale = 0.05
    # q-c
    # draw_one_trajectory(
    #     file_path=base_path + 'q_c_rrcc_5_big.csv',
    #     save=True,
    #     save_name='analysis/data/q-c/q_c_rrcc_5_big.eps',
    #     title='Q-Consensus with noise-scale=0.05'
    # )

    # rl-based
    draw_one_trajectory(
        file_path=base_path + 'rl_rrcc_5_big.csv',
        save=True,
        save_name='analysis/data/q-c/rl_rrcc_5_big.eps',
        title='RL based algorithm with noise-scale=0.05'
    )

    # draw_one_trajectory(
    #     file_path=base_path + rl_noise_path + rl_noise_file_path_2[4],
    #     save=True,
    #     save_name='analysis/data/q-c/rl_rrcc_5.eps',
    #     title='RL based algorithm with noise-scale=0.05'
    # )
