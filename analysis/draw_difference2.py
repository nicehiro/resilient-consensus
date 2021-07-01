import matplotlib.pyplot as plt
from typing import List
import pandas as pd


def draw_difference(path: List[str], fig_title, save_name):
    # matplotlib.use("Agg")
    plt.style.use(["science", "ieee"])
    sub_fig_titles = ["RRRR", "RRCC", "CCCC"] * 3
    colors = [
        "#FFC300",
        "#2ECC71",
        "#9B59B6",
        "#E74C3C",
        "#DAF7A6",
        "#FFC300",
        "#3498DB",
        "#234567",
    ]
    line_labels = ["Node{0}".format(i) for i in range(12)]
    lines = []
    axs_x, axs_y = 3, 3
    fig, axs = plt.subplots(axs_x, axs_y, figsize=[2, 2])
    fig.tight_layout()
    fig.subplots_adjust(
        left=0.1, bottom=0.168, right=0.982, top=0.884, wspace=0.442, hspace=0.855
    )
    fig.suptitle(fig_title)
    for j in range(len(path)):
        df = pd.read_csv(path[j])
        axs[j // axs_y][j % axs_y].set_title(sub_fig_titles[j])
        axs[j // axs_y][j % axs_y].set_xlabel("Times")
        axs[j // axs_y][j % axs_y].set_ylabel("State")
        with plt.style.context(["science", "ieee"]):
            l = axs[j // axs_y][j % axs_y].plot(
                df.iloc[:, 5:].max(axis=1) - df.iloc[:, 5:].min(axis=1),
                linestyle="-",
                linewidth=0.3,
                color="#234567",
            )
            lines.append(l)
    # axs[1][1].legend(labels=line_labels, ncol=2, labelspacing=0.2)
    fig.legend(labels=line_labels, ncol=6, loc="lower center")
    plt.show()
    # plt.savefig(save_name, format="eps")


if __name__ == "__main__":
    base_path = "./analysis/data/q-c/"
    fixed_path = "fixed/"
    switching_path = "switcher/"
    # file_path = ["q_c_rrrr_5_0.csv", "q_c_rrcc_5_0.csv", "q_c_cccc_5_0.csv"] + \
    #             ["q_c_rrrr_5_5.csv", "q_c_rrcc_5_5.csv", "q_c_cccc_5_5.csv"] + \
    #             ["q_c_rrrr_5_10.csv", "q_c_rrcc_5_10.csv", "q_c_cccc_5_10.csv"]
    # file_path = ['q_c_rrrr_10.csv', 'q_c_rrcc_10.csv', 'q_c_cccc_10.csv'] + \
    #             ['q_c_rrrr_9.csv', 'q_c_rrcc_9.csv', 'q_c_cccc_9.csv'] + \
    #             ['q_c_rrrr_8.csv', 'q_c_rrcc_8.csv', 'q_c_cccc_8.csv']
    file_path = (
        [
            "q_c_switcher_rrrr_9.csv",
            "q_c_switcher_rrcc_9.csv",
            "q_c_switcher_cccc_9.csv",
        ]
        + [
            "q_c_switcher_rrrr_6.csv",
            "q_c_switcher_rrcc_6.csv",
            "q_c_switcher_cccc_6.csv",
        ]
        + [
            "q_c_switcher_rrrr_3.csv",
            "q_c_switcher_rrcc_3.csv",
            "q_c_switcher_cccc_3.csv",
        ]
    )
    draw_difference(
        list(map(lambda x: base_path + switching_path + x, file_path)),
        fig_title="Q Consensus for Switching topology",
        save_name="q_c_fixed.eps",
    )
