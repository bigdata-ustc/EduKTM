# coding: utf-8
# 2021/8/6 @ zengxiaonan

import pytest
from EduKTM import AKT


@pytest.mark.parametrize("maxgradnorm", [-1, 1])
@pytest.mark.parametrize("separate_qa", [False, True])
@pytest.mark.parametrize("kq_same", [0, 1])
def test_train(data, conf, tmp_path, maxgradnorm, separate_qa, kq_same):
    n_question, n_pid, batch_size = conf
    n_blocks = 1
    d_model = 32
    dropout = 0.05
    l2 = 1e-5

    akt = AKT(n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size, maxgradnorm, separate_qa)
    akt.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dkt+.params"
    akt.save(filepath)
    akt.load(filepath)
