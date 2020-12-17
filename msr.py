from env import Env, Property
import random

from torch.utils.tensorboard import SummaryWriter


def arc_p(**kwargs):
    writer = SummaryWriter("./logs/arc_p/")
    env = Env(
        nodes_n=10,
        reset_env=kwargs["reset_env"],
        evil_nodes_type=kwargs["evil_nodes_type"],
        times=1,
        with_noise=kwargs["with_noise"],
        directed_graph=kwargs["directed_graph"],
    )
    F = env.nodes_n - env.goods_n
    timesteps = kwargs["timesteps"]
    for t in range(timesteps):
        for i, node in enumerate(env.map.nodes):
            # don't update if node's neighbors <= 2F
            if node.property == Property.GOOD:
                if len(node.weights) - 1 <= 2 * F:
                    continue
                keys = node.weights.keys()
                neigh_i = []
                for k in keys:
                    if k != i:
                        neigh_i.append(env.map.nodes[k].v)
                neigh_i.sort()
                cliped_neighbors = neigh_i[F:-F]
                n = 0
                for v in cliped_neighbors:
                    n += v - node.v
                node.v += n
            elif node.property == Property.CONSTANT:
                pass
            elif node.property == Property.RANDOM:
                node.v = random.random()
        writer.add_scalars(
            "Value of nodes",
            {"Node {0}".format(i): env.map.nodes[i].v for i in range(env.nodes_n)},
            t,
        )


if __name__ == "__main__":
    arc_p(
        timesteps=100,
        reset_env=True,
        evil_nodes_type="3r",
        with_noise=True,
        directed_graph=True,
    )
