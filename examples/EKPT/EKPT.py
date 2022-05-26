# coding: utf-8
# 2021/4/5 @ liujiayu
import logging
import numpy as np
import json
from EduKTM import EKPT

# Q matrix
q_m = np.loadtxt("../../data/2009_skill_builder_data_corrected/q_m.csv", dtype=int, delimiter=",")
prob_num, know_num = q_m.shape[0], q_m.shape[1]

# training data
with open("../../data/2009_skill_builder_data_corrected/train_data.json", encoding='utf-8') as file:
    train_set = json.load(file)
stu_num = max([x['user_id'] for x in train_set[0]]) + 1
time_window_num = len(train_set)
                    
# testing data
with open("../../data/2009_skill_builder_data_corrected/test_data.json", encoding='utf-8') as file:
    test_set = json.load(file)

logging.getLogger().setLevel(logging.INFO)

cdm = EKPT(q_m, stu_num, prob_num, know_num, time_window_num=time_window_num)

cdm.train(train_set, epoch=2, lr=0.001, lr_b=0.0001, epsilon=1e-3, init_method='mean')
cdm.save("ekpt.params")

cdm.load("ekpt.params")
rmse, mae = cdm.eval(test_set)
print("For EKPT, RMSE: %.6f, MAE: %.6f" % (rmse, mae))