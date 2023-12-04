# coding: utf-8
# 2023/11/21 @ xubihan
import random
import numpy as np
import pytest


@pytest.fixture(scope="package")
def conf():
    batch_size = 16
    n_question = 8
    n_exercise = 32
    q_matrix = np.zeros((n_exercise + 1, n_question + 1)) + 0.1
    for row_id in range(n_exercise + 1):
        rand_idx = random.randint(1, n_question)
        q_matrix[row_id][rand_idx] = 1
    return n_question, n_exercise, q_matrix, batch_size


@pytest.fixture(scope="package", params=[0, 8, 16])
def data(conf, request):
    n_question, n_exercise, q_matrix, batch_size = conf
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
    time = [
        [random.uniform(0, 1) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    attempt = [
        [random.uniform(0, 1) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    hint = [
        [random.uniform(0, 1) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
    data = (np.array(e), np.array(a), np.array(time),
            np.array(attempt), np.array(hint))

    return data
