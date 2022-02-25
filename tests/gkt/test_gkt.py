import pytest
from EduKTM import GKT


def test_train(data, conf, tmp_path):
    ku_num, graph, hidden_num = conf
    mgkt = GKT(ku_num, graph, hidden_num)
    mgkt.train(data, test_data=data, epoch=1)
    filepath = tmp_path / "mgkt.params"
    mgkt.save(filepath)
    mgkt.load(filepath)
