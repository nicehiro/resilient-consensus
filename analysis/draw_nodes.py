import sys

sys.path.append("/home/hiro/Documents/resilient-consensus/")
print(sys.path)


import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from env import Env
from car_adjust import CarEnv


def concat_edges(env: Env):
    edges = []
    for i, node in enumerate(env.map.nodes):
        for k, _ in node.weights.items():
            edges.append((k, i))
    return edges


env = Env(nodes_n=12, evil_nodes_type="4r", directed_graph=True)
# env = CarEnv()

matplotlib.use("Agg")
G = nx.DiGraph()
G.add_nodes_from([x for x in range(env.nodes_n)])
G.add_edges_from(concat_edges(env))
nx.draw(
    G,
    pos=nx.circular_layout(G),
    with_labels=True,
    # edge_color='b',
    node_color=["#99A3A4" for _ in range(env.nodes_n - env.goods_n)]
    + ["green" for _ in range(env.goods_n)],
    # arrowstyle='-',
    node_size=1500,
    font_size=14,
)
plt.show()
plt.savefig("nodes.eps", format="eps")
