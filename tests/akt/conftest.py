# coding: utf-8
# 2021/8/6 @ zengxiaonan
import random
import numpy as np
import pytest


@pytest.fixture(scope="package", params=[0, 100])
def conf(request):
    batch_size = 32
    n_question = 10
    n_pid = request.param
    return n_question, n_pid, batch_size


@pytest.fixture(scope="package", params=[0, 10])
def data(conf, request):
    n_question, n_pid, batch_size = conf
    batch_size += request.param
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
    if n_pid > 0:
        p = [
            [random.randint(1, n_pid) for _ in range(seqlen)]
            for _ in range(batch_size)
        ]
    else:
        p = []
    data = (np.array(q), np.array(qa), np.array(p))

    return data
