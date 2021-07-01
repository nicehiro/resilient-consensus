import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')


df = pd.read_csv('./analysis/data/q-c/switch-noise/success_switch.csv')
sns.set(font='Times New Roman')
df = df[::-1].iloc[:, 1:]
df.index = pd.RangeIndex(start=10, stop=91, step=10)
sns.scatterplot(data=df)
plt.plot(df)
plt.title('Q-Consensus Success Rate in Switching Topology')
plt.xlabel('cp% Connection Probability')
plt.ylabel('Success Rate')
# plt.show()
plt.savefig('./analysis/data/q-c/success_switch.eps')
