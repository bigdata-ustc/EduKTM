# coding: utf-8
# 2021/8/17 @ tongshiwei

import logging
from EduKTM.GKT import etl, extract_graph

from EduKTM.GKT import GKT

if __name__ == '__main__':
    batch_size = 16
    train = etl("../../data/assistment_2009_2010/train.json", batch_size)
    valid = etl("../../data/assistment_2009_2010/test.json", batch_size)
    # test = etl("../../data/a0910c/test.json", batch_size)
    graph = extract_graph(124)

    logging.getLogger().setLevel(logging.INFO)

    gkt = GKT(graph, item_dim=5, state_dim=5)
    gkt.train(train, epoch=2, device="cuda:0")
# gkt.save("gkt.params")

# gkt.load("gkt.params")
# auc, accuracy = gkt.eval(test)
# print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
