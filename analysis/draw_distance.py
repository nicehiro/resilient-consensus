import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from attribute import Attribute
from rcenv import Env
from utils import adjacent_matrix


base_path = './data/'
file_path = ['q_c_rrrr.csv', 'q_c_rrcc.csv', 'q_c_ccrr.csv', 'q_c_cccc.csv']


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

# linestyles = ['-', '--', '-.', ':']


fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w')




for i in range(4):
    df = pd.read_csv(base_path + file_path[i])
    env = Env(adjacent_matrix, node_attrs=bad_node_attrs[i] + [Attribute.NORMAL] * 8)
    res = pd.Series(0, index=np.arange(len(df)))
    for j in range(5, 13):
        for k in range(j, 13):
            res += (df.iloc[:, j] - df.iloc[:, k]).pow(2)
    n = 8 * (8 - 1) / 2
    res = res / n
    res = res.pow(1/2)
    res = res[0: 1000]
    res = res[res.index % 10 == 0]

    ax.plot(res)
    ax2.plot(res)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylim(0.4, 0.5)
    ax2.set_ylim(0, 0.2)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax.set_title('Q-Consensus: Convergence index of normal nodes.')
    ax2.set_xlabel('Times')
    fig.text(0.00, 0.5, 'Convergence index', va='center', rotation='vertical')

ax.legend(['RRRR', 'RRCC', 'CCRR', 'CCCC'])

plt.show()
plt.savefig('large-net.eps', format='eps')
