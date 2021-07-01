import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')


df = pd.read_csv('./analysis/data/q-c/fixed-noise/mean_epi_fixed.csv')
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

df = df[::-1].iloc[1:, 1:]
df.index = pd.RangeIndex(start=10, stop=101, step=10)
sns.scatterplot(data=df)
plt.plot(df)
# plt.title('Q-Consensus Convergence Timesteps in Fixed Topology')
plt.xlabel('Faulty Probability')
plt.ylabel('Timesteps')

plt.subplots_adjust(left=0.140, right=0.95, bottom=0.155, top=0.95)
# plt.show()
plt.savefig('./analysis/data/q-c/mean_epi_fixed.eps')
