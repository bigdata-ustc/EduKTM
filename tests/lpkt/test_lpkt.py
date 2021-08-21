# coding: utf-8
# 2021/8/21 @ zengxiaonan

from EduKTM import LPKT


def test_train(data, conf, tmp_path):
    n_at, n_it, n_question, n_exercise, q_matrix, batch_size = conf
    d_a = 16
    d_e = 32
    d_k = 32
    dropout = 0.2

    lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
    lpkt.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "lpkt.params"
    lpkt.save(filepath)
    lpkt.load(filepath)
