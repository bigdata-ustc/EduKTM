# coding: utf-8
# 2022/3/18 @ ouyangjie


import random
import pytest
import numpy as np


@pytest.fixture(scope="package")
def conf():
    n_question = 10
    batch_size = 4
    return n_question, batch_size


@pytest.fixture(scope="package")
def data(conf):
    n_question, batch_size = conf
    seqlen = 10

    q = [
        [random.randint(1, n_question) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    a = [
        [random.randint(0, 1) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    qa = [
        [a[i][j] * n_question + q_id for j, q_id in enumerate(q_seq)]
        for i, q_seq in enumerate(q)
    ]
    data = (np.array(q), np.array(qa))

    return data
