# coding: utf-8
# 2021/9/27 @ zengxiaonan
import random
import numpy as np
import pytest


@pytest.fixture(scope="package", params=[0, 32])
def conf(request):
    batch_size = 16
    n_at = request.param
    n_it = 32
    n_skill = 8
    n_exercise = 32

    return n_at, n_it, n_skill, n_exercise, batch_size


@pytest.fixture(scope="package")
def data(conf):
    n_at, n_it, n_skill, n_exercise, batch_size = conf
    seqlen = 10

    s = [
        [random.randint(1, n_skill) for _ in range(seqlen)]
        for _ in range(batch_size)
    ]
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
    at = None
    if n_at != 0:
        at = [
            [random.randint(1, n_at) for _ in range(seqlen)]
            for _ in range(batch_size)
        ]

    if n_at != 0:
        data = (np.array(s), np.array(a), np.array(e), np.array(it), np.array(at))
    else:
        data = (np.array(s), np.array(a), np.array(e), np.array(it))

    return data
