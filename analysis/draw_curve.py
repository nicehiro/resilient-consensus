import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from typing import List
from env import Env, Property


def draw_trajectory(path: List[str], fig_title, save_name):
    matplotlib.use('Agg')
    evil_nodes_types = ['3r', '2r1c', '1r2c', '3c']
    sub_fig_titles = ['3 Random', '2 Random 1 Constant', '1 Random 2 Constant', '3 Constant']
    colors = ['0.25', '0.25', '0.25', '#FFC300', '#2ECC71', '#9B59B6', '#E74C3C', '#DAF7A6', '#FFC300', '#3498DB']
    line_labels = ['Node{0}'.format(i) for i in range(10)]
    lines = []
    if len(path) != 4:
        raise ValueError('Path should contains 4 sub path.')
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(fig_title)
    for j in range(len(path)):
        env = Env(nodes_n=10, evil_nodes_type=evil_nodes_types[j])
        df = pd.read_csv(path[j])
        axs[j // 2][j % 2].set_title(sub_fig_titles[j])
        axs[j // 2][j % 2].set_xlabel('Times')
        axs[j // 2][j % 2].set_ylabel('State')
        for i in range(1, 11):
            if env.map.nodes[i-1].property is Property.RANDOM:
                colors[i-1] = '0.8'
            elif env.map.nodes[i-1].property is Property.CONSTANT:
                colors[i-1] = '0'
            alpha = 1 if i < 4 else 1.0
            l = axs[j // 2][j % 2].plot(df.iloc[:, i], color=colors[i-1], alpha=alpha)
            lines.append(l)
    axs[1][1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), labels=line_labels, fontsize=8, ncol=1)
    # fig.legend(labels=line_labels, loc='center', ncol=5, bbox_to_anchor=[0.5, 0], bbox_transform=fig.transFigure)
    # bbox_to_anchor = (1.05, 0.3),
    # fig.legend(lines,
    #           bbox_to_anchor=(1.2, 0.5),
    #            borderaxespad=0.,
    #           labels=line_labels,
    #           # loc='center right',
    #           fontsize=8)
    plt.show()
    plt.savefig(save_name, format='eps')


if __name__ == '__main__':
    base_path = './data/'
    file_path = ['q_new_3r.csv', 'q_new_2r1c.csv', 'q_new_1r2c.csv', 'q_new_3c.csv']
    file_path_noise = ['q_new_noise_3r.csv', 'q_new_noise_2r1c.csv', 'q_new_noise_1r2c.csv', 'q_new_noise_3c.csv']
    file_path_ddpg = ['ddpg_3r.csv', 'ddpg_2r1c.csv', 'ddpg_1r2c.csv', 'ddpg_3c.csv']
    file_path_ddpg_noise = ['ddpg_noise_3r.csv', 'ddpg_noise_2r1c.csv', 'ddpg_noise_1r2c.csv', 'ddpg_noise_3c.csv']

    file_path_directed = ['q_new_directed_3r.csv', 'q_new_directed_2r1c.csv', 'q_new_directed_1r2c.csv', 'q_new_directed_3c.csv']
    file_path_directed_noise = ['q_new_noise_directed_3r.csv', 'q_new_noise_directed_2r1c.csv', 'q_new_noise_directed_1r2c.csv', 'q_new_noise_directed_3c.csv']
    draw_trajectory(list(map(lambda x: base_path + x, file_path_noise)),
                    fig_title='Q Consensus',
                    save_name='q_new_trajectories_with_noise.eps')
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
