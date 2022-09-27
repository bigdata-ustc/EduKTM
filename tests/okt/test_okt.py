# coding: utf-8
# 2021/9/27 @ zengxiaonan

from EduKTM import OKT


def test_train(data, conf, tmp_path):
    n_at, n_it, n_question, n_exercise, batch_size = conf
    d_e = 32
    d_q = 16
    d_a = 16
    d_at = 10
    d_p = 16
    d_h = 32
    dropout = 0.2

    okt = OKT(n_at, n_it, n_exercise, n_question, d_e, d_q, d_a, d_at, d_p, d_h,
              batch_size=batch_size, dropout=dropout)
    filepath = tmp_path / "okt.params"
    okt.train(data, test_data=data, epoch=2, filepath=filepath)
    okt.load(filepath)
