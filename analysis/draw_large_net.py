import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from env import Env
from large_env import LargeNet


def concat_edges(env):
    edges = []
    for i, node in enumerate(env.map.nodes):
        for k, _ in node.weights.items():
            edges.append((k, i))
    return edges


# env = Env(nodes_n=10, evil_nodes_type='3r')
env = LargeNet()

# matplotlib.use('Agg')
G = nx.Graph()
G.add_nodes_from([x for x in range(env.nodes_n)])
edges = concat_edges(env)
G.add_edges_from(edges)
# nx.draw(G,
#         # pos=nx.circular_layout(G),
#         # with_labels=True,
#         # edge_color='b',
#         # node_color=['#99A3A4' for _ in range(env.nodes_n-env.goods_n)] + ['green' for _ in range(env.goods_n)],
#         arrowstyle='-',
#         # node_size=1500,
#         node_size=80,
#         # font_size=14
#         # cmap=plt.cm.Reds_r,
#         edge_alpha=0.4,
#         )


# pos = nx.random_layout(G)
pos = nx.spring_layout(G)
dmin, ncenter = 1, 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

p = {}

for i, node in enumerate(env.map.nodes):
    p[i] = len(node.weights)
max_p = max(p.values())
for k, v in p.items():
    p[k] /= max_p


plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(G, pos, nodelist=[ncenter], edge_color='#A0A0A0', width=1)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
)

# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
plt.axis("off")
# plt.title('Large Scale Net Topology')

plt.show()
# plt.savefig('nodes.eps', format='eps')
