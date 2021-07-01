import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from analysis.save import save_nodes_value
import glob
import os
import pprint
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

adjacent_matrix = [
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
]


linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 1))),
    # ('loosely dashed',        (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    # ('densely dashed',        (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


def summary_writer(log_path="log"):
    # summary writer for tensorboard
    return SummaryWriter("logs")


def normalize(data):
    """Normalize data."""
    data = np.array(data, dtype=np.float32)
    mean = data.mean()
    std = np.std(data)
    return (data - mean) / std


def batch_train(train, method="DQN", label="1r2c", n=1000):
    """Train a batch of episode, when they convergence or
    train steps get a certain number stop train.

    @:param train
        a method for training
    @:param method
        train method description
    @:param label
        situation
    @:param n
        train times
    """
    failed_n = 0
    success_n = 0
    for i in range(n):
        env = train()
        if env.is_done():
            success_n += 1
        else:
            failed_n += 1
        print(env.map.node_val())
        print(
            "Test Numbers: {0}\nSuccess Numbers: {1}\tFailed Numbers: {2}".format(
                n, success_n, failed_n
            )
        )
        save_nodes_value(
            env.map, method, env.is_done(), label, path="data/nodes_values.csv"
        )


def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    """Convert all event to pandas dataframe."""
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def logs2df(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str):
    """This is a enhanced version of
    https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b
    This script exctracts variables from all logs from tensorflow event
    files ("event*"),
    writes them to Pandas and finally stores them a csv-file or
    pickle-file including all (readable) runs of the logging directory.
    Example usage:
    # create csv file from all tensorflow logs in provided directory (.)
    # and write it to folder "./converted"
    tflogs2pandas.py . --write-csv --no-write-pkl --o converted
    # creaste csv file from tensorflow logfile only and write into
    # and write it to folder "./converted"
    tflogs2pandas.py tflog.hostname.12345 --write-csv --no-write-pkl --o converted
    """
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(logdir_or_logfile + "/**/event*")
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        pp.pprint(event_paths)
        loss_dfs = {}
        weight_dfs = {}
        for path in event_paths[:]:
            name = path.split("/")[-2]
            if name.split(" ")[0] == "Loss":
                continue
            # Actions of Node 3_Adj 8
            node_i = name.split(" ")[3].split("_")[0]
            adj_i = name.split(" ")[-1]
            if node_i not in weight_dfs:
                weight_dfs[node_i] = {}
            weight_dfs[node_i][adj_i] = tflog2pandas(path)
        return weight_dfs
    else:
        print("No event paths have been found.")
        return None
