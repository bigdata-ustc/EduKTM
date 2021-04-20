# coding: utf-8
# 2021/4/5 @ liujiayu
from EduKTM import KPT


def test_train(data, tmp_path):
    train_set, q_m, stu_num, prob_num, know_num, time_window_num = data

    cdm = KPT('KPT', q_m, stu_num, prob_num, know_num, time_window_num=time_window_num)

    cdm.train(train_set, epoch=30, lr=0.001, lr_b=0.0001, epsilon=1e-1, init_method='mean')
    rmse, mae = cdm.eval([{'user_id': 0, 'item_id': 0, 'score': 1.0}])
    filepath = tmp_path / "kpt.params"
    cdm.save(filepath)
    cdm.load(filepath)

    cdm2 = KPT('EKPT', q_m, stu_num, prob_num, know_num, time_window_num=time_window_num)

    cdm2.train(train_set, epoch=30, lr=0.001, lr_b=0.0001, epsilon=1e-1, init_method='mean')
    rmse2, mae2 = cdm2.eval([{'user_id': 0, 'item_id': 0, 'score': 1.0}])
    filepath2 = tmp_path / "ekpt.params"
    cdm2.save(filepath2)
    cdm2.load(filepath2)
