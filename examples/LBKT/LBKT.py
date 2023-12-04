# coding: utf-8
# 2023/11/21 @ xubihan
import numpy as np
from load_data import DATA
from EduKTM import LBKT
import logging


def generate_q_matrix(path, n_skill, n_problem, gamma=0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix


n_question = 123
memory_size = n_question + 1
n_exercises = 17751

seqlen = 100
dim_tp = 128
num_resps = 2
num_units = 128
dropout = 0.2
dim_hidden = 50
batch_size = 8
q_gamma = 0.1

dat = DATA(seqlen=seqlen, separate_char=',')
data_path = '../../data/2009_skill_builder_data_corrected/'
train_data = dat.load_data(data_path + 'train.txt')
test_data = dat.load_data(data_path + 'test.txt')
q_matrix = generate_q_matrix(
    data_path + 'problem2skill',
    n_question, n_exercises,
    q_gamma
)

logging.getLogger().setLevel(logging.INFO)

lbkt = LBKT(n_exercises, dim_tp, num_resps, num_units, dropout,
            dim_hidden, memory_size, batch_size, q_matrix)
lbkt.train(train_data, test_data, epoch=2, lr=0.001)
lbkt.save("lbkt.params")

lbkt.load("lbkt.params")
_, auc, accuracy, rmse = lbkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, accuracy, rmse))
