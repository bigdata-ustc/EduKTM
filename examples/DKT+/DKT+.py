# coding: utf-8
# 2021/5/26 @ tongshiwei
import logging
from EduKTM.DKTPlus import etl

from EduKTM import DKTPlus

batch_size = 64
train = etl("../../data/a0910c/train.json", batch_size)
valid = etl("../../data/a0910c/valid.json", batch_size)
test = etl("../../data/a0910c/test.json", batch_size)

logging.getLogger().setLevel(logging.INFO)

dkt_plus = DKTPlus(ku_num=146, hidden_num=100, loss_params={"lr": 0.1, "lw1": 0.5, "lw2": 0.5})
dkt_plus.train(train, valid, epoch=2)
dkt_plus.save("dkt+.params")

dkt_plus.load("dkt+.params")
auc, accuracy = dkt_plus.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
