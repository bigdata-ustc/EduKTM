# coding: utf-8
# 2021/8/6 @ zengxiaonan

from EduKTM import AKT


def test_train(data, conf, tmp_path):
    n_question, n_pid, batch_size = conf
    n_blocks = 1
    d_model = 32
    dropout = 0.05
    kq_same = 1
    l2 = 1e-5
    maxgradnorm = -1
    akt = AKT(n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size, maxgradnorm)
    akt.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dkt+.params"
    akt.save(filepath)
    akt.load(filepath)
