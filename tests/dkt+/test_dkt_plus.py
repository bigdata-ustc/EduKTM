# coding: utf-8
# 2021/5/26 @ tongshiwei

import pytest
from EduKTM import DKTPlus


@pytest.mark.parametrize("lr", [0, 0.1])
@pytest.mark.parametrize("lw1", [0, 0.5])
@pytest.mark.parametrize("lw2", [0, 0.5])
@pytest.mark.parametrize("add_embedding_layer", [True, False])
def test_train(data, conf, tmp_path, lr, lw1, lw2, add_embedding_layer):
    ku_num, hidden_num = conf
    dkt_plus = DKTPlus(
        ku_num, hidden_num,
        net_params={"add_embedding_layer": add_embedding_layer},
        loss_params={"lr": lr, "lw1": lw1, "lw2": lw2}
    )
    dkt_plus.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dkt+.params"
    dkt_plus.save(filepath)
    dkt_plus.load(filepath)
