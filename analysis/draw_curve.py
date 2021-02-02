import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from typing import List
from rcenv import Env
from attribute import Attribute
from utils import adjacent_matrix


def draw_trajectory(path: List[str], fig_title, save_name):
    # matplotlib.use('Agg')
    plt.style.use(['science', 'ieee'])
    bad_node_attrs = [
        [Attribute.RANDOM] * 4,
        [Attribute.RANDOM] * 2 + [Attribute.CONSTANT] * 2,
        [Attribute.CONSTANT] * 2 + [Attribute.RANDOM] * 2,
        [Attribute.CONSTANT] * 4
    ]
    evil_nodes_types = ['rrrr', 'rrcc', 'ccrr', 'cccc']
    sub_fig_titles = ['RRRR', 'RRCC', 'CCRR', 'CCCC']
    colors = ['0.25', '0.25', '0.25', '#FFC300', '#2ECC71', '#9B59B6', '#E74C3C', '#DAF7A6', '#FFC300', '#3498DB',
              '#999999', '#111111']
    line_labels = ['Node{0}'.format(i) for i in range(12)]
    lines = []
    if len(path) != 4:
        raise ValueError('Path should contains 4 sub path.')
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(fig_title)
    for j in range(len(path)):
        node_attrs = bad_node_attrs[j] + [Attribute.NORMAL] * 8
        env = Env(adj_matrix=adjacent_matrix, node_attrs=node_attrs)
        df = pd.read_csv(path[j])
        axs[j // 2][j % 2].set_title(sub_fig_titles[j])
        axs[j // 2][j % 2].set_xlabel('Times')
        axs[j // 2][j % 2].set_ylabel('State')
        # with plt.style.context(['science', 'light']):
        #     for i in range(1, 5):
        #         l = axs[j // 2][j % 2].plot(df.iloc[:, i])
        #         lines.append(l)
        with plt.style.context(['science', 'ieee']):
            for i in range(5, 13):
                l = axs[j // 2][j % 2].plot(df[df.iloc[:, 0] % 40 == 0].iloc[:, i])
                lines.append(l)
    axs[1][1].legend(labels=line_labels)
    # axs[1][1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), labels=line_labels, fontsize=8, ncol=1)
    # fig.legend(labels=line_labels, loc='center', ncol=5, bbox_to_anchor=[0.5, 0], bbox_transform=fig.transFigure)
    # bbox_to_anchor = (1.05, 0.3),
    # fig.legend(lines,
    #           bbox_to_anchor=(1.2, 0.5),
    #            borderaxespad=0.,
    #           labels=line_labels,
    #           # loc='center right',
    #           fontsize=8)
    plt.show()
    # plt.savefig(save_name, format='eps')


if __name__ == '__main__':
    base_path = './data/'
    file_path = ['q_c_rrrr.csv', 'q_c_rrcc.csv', 'q_c_ccrr.csv', 'q_c_cccc.csv']
    draw_trajectory(list(map(lambda x: base_path + x, file_path)),
                    fig_title='Q Consensus',
                    save_name='q_consensus_trajectories.eps')
    # draw_trajectory(list(map(lambda x: base_path + x, file_path)),
    #                 fig_title='Q Consensus(5% tolerance, no noise)',
    #                 save_name='q_new_trajectories.eps')
    # draw_trajectory(list(map(lambda x: base_path + x, file_path_ddpg)),
    #                 fig_title='DDPG(5% tolerance, no noise)',
    #                 save_name='ddpg_trajectories.eps')
    # draw_trajectory(list(map(lambda x: base_path + x, file_path_ddpg_noise)),
    #                 fig_title='D-DDPG',
    #                 save_name='ddpg_trajectories_with_noise.eps')
    # draw_trajectory(list(map(lambda x: base_path + x, file_path_directed)),
    #                 fig_title='Q Consensus Directed Graph\n(5% tolerance, no noise)',
    #                 save_name='q_new_directed_trajectories.eps')
    # draw_trajectory(list(map(lambda x: base_path + x, file_path_directed_noise)),
    #                 fig_title='Q Consensus Directed Graph',
    #                 save_name='q_new_noise_directed_trajectories.eps')
