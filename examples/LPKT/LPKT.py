import logging
import numpy as np
from load_data import DATA
from EduKTM import LPKT

def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix

batch_size = 64
n_at = 9632
n_it = 2890
n_question = 102
n_exercise = 3162
seqlen = 500
d_k = 128
d_a = 50
d_e = 128
q_gamma = 0.3
dropout = 0.2

dat = DATA(seqlen=seqlen, separate_char=',')
train_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/train.txt')
test_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/test.txt')
q_matrix = generate_q_matrix(
    '../../data/anonymized_full_release_competition_dataset/problem2skill',
    n_question, n_exercise,
    q_gamma
)

logging.getLogger().setLevel(logging.INFO)

lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
lpkt.train(train_data, test_data, epoch=2)
lpkt.save("lpkt.params")

lpkt.load("lpkt.params")
_, auc, accuracy = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
