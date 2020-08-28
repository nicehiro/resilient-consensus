import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use('Agg')

fig, ax = plt.subplots(1, 1, constrained_layout=True)

df = pd.read_csv('./data/large_net.csv')

ax.plot(df['Step'][1:], df['Value'][1:])
ax.set_title('Large Scale Net')
ax.set_xlabel('Times')
ax.set_ylabel('Distances of All Good Nodes')

plt.show()
plt.savefig('large-net.eps', format='eps')