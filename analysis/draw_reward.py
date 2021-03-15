import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as tick


def x_fmt(tick_val, pos):
    val = int(tick_val) / 10000
    return '{0}'.format(val)


base_path = './data/ddpg-vs-ddpg/'

matplotlib.use('Agg')

font = {'size': 14,
        'weight': 'bold'}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=[6,3])

for file_name in os.listdir(base_path):
    if file_name.startswith('.'):
        continue
    if os.path.isdir(os.path.join(base_path, file_name)):
        continue
    df = pd.read_csv(os.path.join(base_path, file_name))
    ax.plot(df['Step'], df['Value'])
ax.legend(['Node 2', 'Node 3'], fontsize=10)
ax.set_title('D-DDPG vs D-DDPG Rewards')
ax.set_xlabel('Times / Million')
ax.set_ylabel('Rewards')
ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))

plt.subplots_adjust(bottom=0.17)
plt.show()
plt.savefig('ddpg-vs-ddpg.eps', format=('eps'))