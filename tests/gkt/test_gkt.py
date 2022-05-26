# coding: utf-8
# 2022/2/25 @ fannazya
from EduKTM import GKT


def test_train(data, conf, graph_file, tmp_path):
    ku_num, hidden_num = conf
    mgkt = GKT(ku_num, graph_file, hidden_num)
    mgkt.train(data, test_data=data, epoch=1)
    filepath = tmp_path / "mgkt.params"
    mgkt.save(filepath)
    mgkt.load(filepath)
