import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from rcenv import Env
from utils import adjacent_matrix
from attribute import Attribute
from colors import Color
import random


bads_n = 4
goods_n = 8
node_attrs = [Attribute.RANDOM] * 4 + [Attribute.NORMAL] * 8

probs = [0.5] * 4 + [0.5] * 8
seeds = [random.random() * 100 for _ in range(12)]
env = Env(adjacent_matrix, node_attrs, probs=probs, seeds=seeds)

# matplotlib.use("Agg")
G = nx.DiGraph()
G.add_nodes_from([(i, {'color': Color.GREEN.value if node_attrs[i] is Attribute.NORMAL else Color.RED.value})
                  for i in range(env.n)])

options = {'node_size': 1000}
pos = nx.circular_layout(G)
# draw nodes
nx.draw_networkx_nodes(G, pos=pos,
                       nodelist=[0, 1, 2, 3], node_color=Color.RED.value, **options)
nx.draw_networkx_nodes(G, pos=pos,
                       nodelist=[i for i in range(4, 12)], node_color=Color.GREEN.value, **options)
# draw node label
nx.draw_networkx_labels(G, pos)

# draw edges
for i, node in enumerate(env.topology.nodes):
    for adj, w in node.weights.items():
        color = Color.GREEN if adj.attribute is Attribute.NORMAL else Color.RED
        style = 'solid' if adj.attribute is Attribute.NORMAL else 'dashed'
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist=[[adj.index, i]],
                               width=1,
                               style=style,
                               arrows=True,
                               arrowsize=10,
                               edge_color=color.value,
                               **options)

plt.savefig("nodes.png", format="png")
