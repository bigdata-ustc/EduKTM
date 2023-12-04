# coding: utf-8
# 2023/11/21 @ xubihan

from EduKTM import LBKT


def test_train(data, conf, tmp_path):
    n_question, n_exercise, q_matrix, batch_size = conf
    dim_hidden = 16
    num_resps = 2
    num_units = 32
    dim_tp = 32
    dropout = 0.2
    memory_size = n_question + 1

    lbkt = LBKT(n_exercise, dim_tp, num_resps, num_units,
                dropout, dim_hidden, memory_size, batch_size, q_matrix)
    lbkt.train(data, test_data=data, epoch=2, lr=0.001)
    filepath = tmp_path / "lbkt.params"
    lbkt.save(filepath)
    lbkt.load(filepath)
