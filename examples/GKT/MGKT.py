# coding: utf-8
# 2022/2/25 @ fannazya


import logging
from EduKTM.GKT import etl
from EduKTM import GKT

batch_size = 16
train = etl("../../data/assistment_2009_2010/train.json", batch_size=batch_size)
valid = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)
test = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)

logging.getLogger().setLevel(logging.INFO)

model = GKT(ku_num=124, graph="../../data/assistment_2009_2010/transition_graph.json", hidden_num=5)
model.train(train, valid, epoch=2)
model.save("mgkt.params")

model.load("mgkt.params")
auc, accuracy = model.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
