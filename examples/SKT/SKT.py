# coding: utf-8

import logging
from EduKTM.GKT import etl
from EduKTM import SKT
import torch

batch_size = 16
train = etl("../../data/assistment_2009_2010/train.json",
            batch_size=batch_size)
valid = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)
test = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)

logging.getLogger().setLevel(logging.INFO)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = SKT(ku_num=124, graph_params=[
            ['../../data/assistment_2009_2010/correct_transition_graph.json', True],
            ['../../data/assistment_2009_2010/ctrans_sim.json', False]
            ], hidden_num=5)
model.train(train, valid, epoch=2, device=device)
model.save("skt.params")

model.load("skt.params")
auc, accuracy = model.eval(test, device=device)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
