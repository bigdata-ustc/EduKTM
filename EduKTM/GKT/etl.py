# coding: utf-8
# 2021/5/24 @ tongshiwei

import dgl
import json
from tqdm import tqdm
import networkx as nx
from EduKTM.utils.torch_utils import PadSequence, FixedBucketSampler
import torch


def extract(data_src):
    """

    Parameters
    ----------
    data_src

    Returns
    -------
    responses: list of tuple
        [(item_id, 0/1)]

    """
    responses = []
    step = 200
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)
            for i in range(0, len(data), step):
                if len(data[i: i + step]) < 2:
                    continue
                responses.append(data[i: i + step])

    return responses


def extract_graph(node_num=None, method="dense", add_self_loop=True, filepath=None):
    """

    Parameters
    ----------
    node_num
    method
    add_self_loop
    filepath

    Returns
    -------

    Examples
    --------
    >>> g = extract_graph(3)
    >>> g.nodes()
    tensor([0, 1, 2])
    >>> g.edges()
    (tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    >>> g.edata["in_a"]
    tensor([[0.0000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.0000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.0000]])
    >>> g.edata["out_a"]
    tensor([[0.0000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.0000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.0000]])
    """
    if method == "dense":
        nx_g = nx.DiGraph()
        for i in range(node_num):
            nx_g.add_edge(i, i, in_a=[0], out_a=[0])
            for j in range(i + 1, node_num):
                weight = [1 / (node_num - 1)]
                nx_g.add_edge(i, j, in_a=weight, out_a=weight)
                nx_g.add_edge(j, i, in_a=weight, out_a=weight)
        return dgl.from_networkx(nx_g, edge_attrs=["in_a", "out_a"])
    else:
        raise NotImplementedError


def transform(raw_data, batch_size, num_buckets=100):
    responses = raw_data

    batch_idxes = FixedBucketSampler([len(rs) for rs in responses], batch_size, num_buckets=num_buckets)
    batch = []

    def index(r):
        correct = 0 if r[1] <= 0 else 1
        return r[0] * 2 + correct

    for batch_idx in tqdm(batch_idxes, "batchify"):
        batch_rs = []
        batch_item_id = []
        batch_next_item_id = []
        batch_labels = []
        for idx in batch_idx:
            batch_item_id.append([r[0] for r in responses[idx]])
            batch_rs.append([index(r) for r in responses[idx]])
            if len(responses[idx]) <= 1:  # pragma: no cover
                next_item_id, labels = [], []
            else:
                next_item_id, labels = zip(*[(r[0], 0 if r[1] <= 0 else 1) for r in responses[idx][1:]])
            batch_next_item_id.append(list(next_item_id))
            batch_labels.append(list(labels))

        max_len = max([len(rs) for rs in batch_rs])
        padder = PadSequence(max_len, pad_val=0)
        batch_rs, data_mask = zip(*[(padder(rs), len(rs)) for rs in batch_rs])
        batch_item_id, data_mask = zip(*[(padder(item_id), len(item_id)) for item_id in batch_item_id])

        max_len = max([len(rs) for rs in batch_labels])
        padder = PadSequence(max_len, pad_val=0)
        batch_labels, label_mask = zip(*[(padder(labels), len(labels)) for labels in batch_labels])
        batch_next_item_id = [padder(next_item_id) for next_item_id in batch_next_item_id]
        # Load
        batch.append(
            [torch.tensor(batch_item_id),
             torch.tensor(batch_rs), torch.tensor(data_mask), torch.tensor(batch_labels),
             torch.tensor(batch_next_item_id),
             torch.tensor(label_mask)])

    return batch


def etl(data_src, batch_size, **kwargs):  # pragma: no cover
    raw_data = extract(data_src)
    return transform(raw_data, batch_size, **kwargs)
