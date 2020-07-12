import os

import pandas as pd

from env import Map


def save_nodes_value(data: Map, method: str, result: bool, label: str, path: str):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['Node{0}'.format(i) for i in range(10)] + ['method', 'success', 'label'])
    else:
        df = pd.read_csv(path, index_col=0)
    index = df.__len__()
    data = {'Node{0}'.format(i): data.nodes[i].v for i in range(10)}
    data['method'] = method
    data['success'] = result
    data['label'] = label
    row = pd.Series(data=data)
    df.loc[index] = row
    df.to_csv(path)
