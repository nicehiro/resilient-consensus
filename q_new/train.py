from env import Env
import math


def make_q(env: Env, init_value):
    q = [{} for _ in range(env.nodes_n)]
    for i, node in enumerate(env.map.nodes):
        for k, v in node.weights.items():
            q[i][k] = init_value
    return q


def q_consensus():
    env = Env(nodes_n=10, times=10)
    Q = make_q(env, 1)
    step_size = 0.1
    print(env.map)
    for epi in range(100):
        for i, node in enumerate(env.map.nodes):
            for j, q in Q[i].items():
                r_ij = math.exp(- abs(env.map.nodes[j].v - node.v) * 10)
                Q[i][j] += step_size * (r_ij - Q[i][j])
            q_sum = sum(Q[i].values())
            for j in Q[i].keys():
                w = (Q[i][j] / q_sum) * (1 - 1 / len(Q[i])) + 0.001
                env.map.update_by_weight(i, j, w)
            env.map.normalize(i)
        env.map.update_value_of_node()
        if epi % 10 == 0:
            print(env.map)


if __name__ == '__main__':
    q_consensus()
