# coding: utf-8
# 2021/5/26 @ tongshiwei

from EduKTM import DKTPlus


def test_train(data, conf, tmp_path):
    ku_num, hidden_size = conf
    dkt_plus = DKTPlus(ku_num, hidden_size)
    dkt_plus.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dkt+.params"
    dkt_plus.save(filepath)
    dkt_plus.load(filepath)
