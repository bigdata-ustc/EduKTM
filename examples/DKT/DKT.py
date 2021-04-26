# coding: utf-8
# 2021/4/24 @ zengxiaonan
import logging
import numpy as np
import torch
import torch.utils.data as Data

from EduKTM import DKT


NUM_QUESTIONS = 123
BATCH_SIZE = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1


def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


train_loader = get_data_loader('../../data/2009_skill_builder_data_corrected/train_data.npy', BATCH_SIZE, True)
test_loader = get_data_loader('../../data/2009_skill_builder_data_corrected/test_data.npy', BATCH_SIZE, False)

logging.getLogger().setLevel(logging.INFO)

dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)
dkt.train(train_loader, epoch=2)
dkt.save("dkt.params")

dkt.load("dkt.params")
auc = dkt.eval(test_loader)
print("auc: %.6f" % auc)
