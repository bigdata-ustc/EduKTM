# coding: utf-8
# 2021/8/21 @ zengxiaonan
import random
import numpy as np
import pytest


@pytest.fixture(scope="package")
def conf():
    batch_size = 16
    n_at = 64
    n_it = 32
    n_question = 8
    n_exercise = 32

    q_matrix = np.zeros((n_exercise + 1, n_question + 1)) + 0.3
    for row_id in range(n_exercise + 1):
        rand_idx = random.randint(1, n_question)
        q_matrix[row_id][rand_idx] = 1
    return n_at, n_it, n_question, n_exercise, q_matrix, batch_size


@pytest.fixture(scope="package", params=[0, 8])
def data(conf, request):
    n_at, n_it, _, n_exercise, _, batch_size = conf
    batch_size += request.param
    seqlen = 10

    a = [
        [random.randint(0, 1) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    e = [
        [random.randint(1, n_exercise) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    it = [
        [random.randint(1, n_it) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    at = [
        [random.randint(1, n_at) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    data = (np.array(a), np.array(e), np.array(it), np.array(at))

    return data
