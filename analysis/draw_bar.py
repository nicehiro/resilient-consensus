import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def draw_bar(ddpg_rates, q_consensus_rates, title):
    matplotlib.use('Agg')

    font = {'size': 14,
            'weight': "normal"}
    matplotlib.rc('font', **font)
    plt.rcParams["font.family"] = "Times New Roman"

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rcParams['text.usetex'] = True

    N = 4

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots(constrained_layout=True, figsize=[6,2])

    ax.set_ylim(90, 102)
    rects1 = ax.bar(ind, ddpg_rates, width, color='orange')

    rects2 = ax.bar(ind + width, q_consensus_rates, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Success Rate %')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('3R', '2R1C', '1R2C', '3C'))
    ax.set_xlabel('Faulty Nodes Types')

    ax.legend((rects1[0], rects2[0]), ('D-DDPG', 'Q Consensus'), loc='lower right', fontsize=10)


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.001*height,
                    height,
                    ha='center', va='bottom', fontsize=12)

    autolabel(rects1)
    autolabel(rects2)

    # plt.subplots_adjust(bottom=0.2)
    plt.show()
    plt.savefig('compare_ddpg_q.eps', format='eps')


if __name__ == '__main__':
    # draw_bar(ddpg_rates=(97.6, 92, 99.8, 99.3),
    #          q_consensus_rates=(100, 100, 98.9, 93.7),
    #          title='(5% tolerance, no noise)')
    draw_bar(ddpg_rates=(98.4, 92.5, 99.9, 99.2),
             q_consensus_rates=(100, 100, 98.8, 95.8),
             title='(5% tolerance, 1% noise)')