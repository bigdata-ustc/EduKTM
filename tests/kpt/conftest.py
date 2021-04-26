# coding: utf-8
# 2021/4/5 @ liujiayu

import random
import numpy as np
import pytest


@pytest.fixture(scope="package")
def conf():
    user_num = 5
    item_num = 2
    know_num = 3
    time_window_num = 2
    return user_num, item_num, know_num, time_window_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num, know_num, time_window_num = conf
    q_m = np.zeros(shape=(item_num, know_num))
    for i in range(item_num):
        q_m[i, random.randint(0, know_num - 1)] = 1

    train_set = []
    for t in range(time_window_num):
        train_t = []
        for u in range(user_num):
            item_t = random.randint(0, item_num - 1)
            rating_t = random.randint(0, 1)
            train_t.append({'user_id': u, 'item_id': item_t, 'score': rating_t})
        train_set.append(train_t)

    return train_set, q_m, user_num, item_num, know_num, time_window_num
