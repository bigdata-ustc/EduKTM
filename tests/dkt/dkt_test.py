# coding: utf-8
# 2021/4/24 @ zengxiaonan

from EduKTM import DKT


def test_train(data, conf, tmp_path):
    num_questions, hidden_size, num_layers = conf
    dkt = DKT(num_questions, hidden_size, num_layers)
    dkt.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dkt.params"
    dkt.save(filepath)
    dkt.load(filepath)
