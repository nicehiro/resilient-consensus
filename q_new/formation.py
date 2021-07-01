from rcenv import Env
import math
import numpy as np
from attribute import Attribute
from typing import List
import json
import os


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.n)]
    for i, node in enumerate(env.topology.nodes):
        for adj, _ in node.weights.items():
            q[i][str(adj.index)] = init_value
    return q


def q_consensus(positions: List[tuple]) -> List[tuple]:
    distances = [
        [(0, 0), (1, 0), (0, -1), (1, -1)],
        [(-1, 0), (1, 0), (-1, -1), (0, -1)],
        [(0, 1), (1, 1), (0, 0), (1, 0)],
        [(-1, 1), (0, 1), (-1, 0), (0, 0)],
    ]
    adjacent_matrix = [[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1]]
    node_attrs = [Attribute.RANDOM] + [Attribute.NORMAL] * 3
    probs = [1.0] * 1 + [1.0] * 3
    seeds = [i for i in range(4)]
    env = Env(
        adjacent_matrix,
        node_attrs=node_attrs,
        probs=probs,
        seeds=seeds,
        times=1,
        noise_scale=0,
    )
    if os.path.exists("q.json"):
        with open("q.json") as json_file:
            q_json = json.load(json_file)
            Q = json.loads(q_json)
    else:
        Q = make_q(env, 1)
    step_size = 0.1
    # set current position
    for i, pos in enumerate(positions):
        env.topology.nodes[i].value = np.array(pos)
    # update weights
    for i, node in enumerate(env.topology.nodes):
        rewards = {}
        for j, q in Q[i].items():
            j = int(j)
            node_j = env.topology.nodes[j]
            dis = np.mean(
                np.abs(node.value - node_j.value - np.array(distances[j][i]))
            ).item()
            r_ij = math.exp(-10 * dis)
            r_ij = node.weights[node_j] * r_ij
            rewards[j] = r_ij
        sum_r = sum(rewards.values())
        for j, q in Q[i].items():
            j = int(j)
            r_ij = rewards[j] / sum_r
            Q[i][str(j)] += max(step_size, 0) * (r_ij - Q[i][str(j)])
        q_sum = sum(Q[i].values())
        for j in Q[i].keys():
            j = int(j)
            w = Q[i][str(j)] / q_sum
            node.update_weight_by_adj(env.topology.nodes[j], w)
    # get next position
    for node in env.topology.nodes:
        m = 0
        for adj, w in node.weights.items():
            # noise = self.noise_scale * (random.random() * 2 - 1) * self.times
            # noise = 0 if not has_noise else (random.random() * 2 - 1) / 100 * self.times
            # w = 0 if w < 0.01 else w
            # w = 0 if w < 0.1 else w
            m += w * (
                adj.value - node.value - np.array(distances[node.index][adj.index])
            )
        node.value = node.value + m

    next_positions = []
    for node in env.topology.nodes:
        t = tuple(node.value)
        next_positions.append(t)

    # save Q
    q_json = json.dumps(Q)
    with open("q.json", "w") as outfile:
        json.dump(q_json, outfile)
    return next_positions


if __name__ == "__main__":
    positions = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for i in range(1000):
        next_positions = q_consensus(positions)
        print(next_positions)
        positions = next_positions
