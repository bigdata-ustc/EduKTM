# coding: utf-8
# 2021/5/25 @ tongshiwei

import torch
import json
from tqdm import tqdm
from EduKTM.utils.torch_utils import PadSequence, FixedBucketSampler


def extract(data_src):  # pragma: no cover
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


def transform(raw_data, batch_size, num_buckets=100):
    # 定义数据转换接口
    # raw_data --> batch_data

    responses = raw_data

    batch_idxes = FixedBucketSampler([len(rs) for rs in responses], batch_size, num_buckets=num_buckets)
    batch = []

    def index(r):
        correct = 0 if r[1] <= 0 else 1
        return r[0] * 2 + correct

    for batch_idx in tqdm(batch_idxes, "batchify"):
        batch_rs = []
        batch_pick_index = []
        batch_labels = []
        for idx in batch_idx:
            batch_rs.append([index(r) for r in responses[idx]])
            if len(responses[idx]) <= 1:  # pragma: no cover
                pick_index, labels = [], []
            else:
                pick_index, labels = zip(*[(r[0], 0 if r[1] <= 0 else 1) for r in responses[idx][1:]])
            batch_pick_index.append(list(pick_index))
            batch_labels.append(list(labels))

        max_len = max([len(rs) for rs in batch_rs])
        padder = PadSequence(max_len, pad_val=0)
        batch_rs, data_mask = zip(*[(padder(rs), len(rs)) for rs in batch_rs])

        max_len = max([len(rs) for rs in batch_labels])
        padder = PadSequence(max_len, pad_val=0)
        batch_labels, label_mask = zip(*[(padder(labels), len(labels)) for labels in batch_labels])
        batch_pick_index = [padder(pick_index) for pick_index in batch_pick_index]
        # Load
        batch.append(
            [torch.tensor(batch_rs), torch.tensor(data_mask), torch.tensor(batch_labels),
             torch.tensor(batch_pick_index),
             torch.tensor(label_mask)])

    return batch


def etl(data_src, batch_size, **kwargs):  # pragma: no cover
    raw_data = extract(data_src)
    return transform(raw_data, batch_size, **kwargs)
