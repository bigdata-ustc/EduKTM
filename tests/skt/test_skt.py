# coding: utf-8
# 2023/3/17 @ weizhehuang0827
from EduKTM import SKT


def test_train(data, conf, graphs, tmp_path):
    ku_num, hidden_num = conf
    mgkt = SKT(ku_num, graphs, hidden_num)
    mgkt.train(data, test_data=data, epoch=1)
    filepath = tmp_path / "skt.params"
    mgkt.save(filepath)
    mgkt.load(filepath)
