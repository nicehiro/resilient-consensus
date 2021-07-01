import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def draw_success_rate(path, title, xlabel, ylabel, save_name, save=True):
    base_path = './analysis/data/q-c/'
    if save:
        matplotlib.use('Agg')
    df = pd.read_csv(base_path + path)
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
    # df = df[::-1].iloc[:, 1:]
    # df.index = pd.RangeIndex(start=10, stop=91, step=10)
    # sns.scatterplot(data=df)
    # plt.plot(df)
    df.plot(x='Baseline', y='RRCC')
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplots_adjust(left=0.140, right=0.95, bottom=0.155, top=0.95)
    if save:
        plt.savefig(base_path + save_name)
    else:
        plt.show()


if __name__ == '__main__':
    # draw_success_rate(
    #     path='mean_epi_baseline.csv',
    #     title='The convergence timesteps in different baseline.',
    #     xlabel='Convergence Condition',
    #     ylabel='Timesteps',
    #     save_name='mean_epi_baseline.eps'
    # )

    # draw_success_rate(
    #     path='mean_epi_noise_scale.csv',
    #     title='The convergence timesteps in different noise scale.',
    #     xlabel='Noise scale',
    #     ylabel='Timesteps',
    #     save_name='mean_epi_noise_scale.eps',
    # )

    # draw_success_rate(
    #     path='success_baseline.csv',
    #     title='The success rate in different baseline.',
    #     xlabel='Convergence Condition',
    #     ylabel='Success Rate',
    #     save=True,
    #     save_name='success_baseline.eps'
    # )

    draw_success_rate(
        path='success_noise_scale.csv',
        title='The success rate in different baseline.',
        xlabel='Noise',
        ylabel='Success Rate',
        save_name='success_noise_scale.eps'
    )
