# coding: utf-8
# 2021/4/23 @ zengxiaonan
import random

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture(scope="package")
def conf():
    num_questions = 10
    hidden_size = 5
    num_layers = 1
    return num_questions, hidden_size, num_layers


@pytest.fixture(scope="package")
def data(conf):
    num_questions, hidden_size, num_layers = conf
    input_size = num_questions * 2
    data = torch.zeros((8, 5, input_size))
    for seq in data:
        for ques in seq:
            one_index = random.randint(0, input_size - 1)
            ques[one_index] = 1
    data = torch.FloatTensor(data)
    batch_size = 4

    # dataset = TensorDataset(torch.FloatTensor(data.tolist()))
    return DataLoader(data, batch_size=batch_size)
