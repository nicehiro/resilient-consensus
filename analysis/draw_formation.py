import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


base_path = "./analysis/data/q-c/formation/"
df_x = pd.read_csv(base_path + "q-c-x.csv")
df_y = pd.read_csv(base_path + "q-c-y.csv")

dis_x = [0, 0, 0, 1]
dis_y = [0, 0, -1, -1]

df_x = df_x.iloc[:, 1:] + dis_x
df_y = df_y.iloc[:, 1:] + dis_y

df_x = df_x.clip(0, 3)
df_y = df_y.clip(0, 3)

# matplotlib.use("Agg")
# sns.set(font='Times New Roman')
plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


names = ['Node0', 'Node1', 'Node2', 'Node3']
colors = [
    '#B71C1C',
    '#424242',
    '#4A148C',
    '#B388FF',
]
markers = ['^', '.', '*', '1']
indexs = np.arange(0, 1000)


for i in range(4):
    ax.scatter(df_x[names[i]][:60], df_y[names[i]][:60], indexs[:60], c=colors[i], marker=markers[i])

xx, yy = np.meshgrid(np.arange(1.5, 3, 0.2), np.arange(0, 1.7, 0.2))
z = np.ones_like(xx) * 60
ax.plot_surface(xx, yy, z, alpha=0.2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Timesteps")

plt.show()
# plt.savefig('formation.eps')