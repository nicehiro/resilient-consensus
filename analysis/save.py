import os

import pandas as pd

from env import Map


def save_nodes_value(data: Map, method: str, path: str):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(10)] + ['method'])
    else:
        df = pd.read_csv(path)
    data = {'Node{0}'.format(i): data.nodes[i].v for i in range(10)}
    data['method'] = method
    row = pd.Series(data=data)
    df.append(data, ignore_index=False)
    df.to_csv(path)
