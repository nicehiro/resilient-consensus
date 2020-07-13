import pandas as pd
import matplotlib.pyplot as plt
import math


def draw_pie(path: str) -> None:
    df = pd.read_csv(path, index_col=0)
    success_data = df.groupby(['label', 'method'])['success']
    total_tests = success_data.size().reset_index(name='Sum')
    success_tests = success_data.sum().reset_index(name='SuccessTimes')
    data = pd.concat([success_tests, total_tests['Sum']], axis=1)
    res = data['SuccessTimes'] / data['Sum']
    res['Title'] = data['method'] + ' ' + data['label']
    plots_n = data.__len__()
    x = math.floor(math.sqrt(plots_n))
    y = math.ceil(plots_n / x)
    fig, axs = plt.subplots(x, y)
    for i in range(plots_n):
        axs[i % y].pie([res[i], 1 - res[i]],
                       labels=['SuccessTimes', 'FailedTimes'],
                       autopct='%.0f%%',
                       shadow=False,
                       radius=0.8)
        axs[i % y].set_title(res['Title'][i])
    plt.show()


if __name__ == '__main__':
    draw_pie('~/Documents/Projects/RL/opinion/data/nodes_values.csv')
