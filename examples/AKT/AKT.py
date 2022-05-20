# coding: utf-8
# 2021/8/5 @ zengxiaonan

from load_data import DATA, PID_DATA
import logging
from EduKTM import AKT

batch_size = 64
model_type = 'pid'
n_question = 123
n_pid = 17751
seqlen = 200
n_blocks = 1
d_model = 256
dropout = 0.05
kq_same = 1
l2 = 1e-5
maxgradnorm = -1

if model_type == 'pid':
    dat = PID_DATA(n_question=n_question, seqlen=seqlen, separate_char=',')
else:
    dat = DATA(n_question=n_question, seqlen=seqlen, separate_char=',')
train_data = dat.load_data('../../data/2009_skill_builder_data_corrected/train_pid.txt')
test_data = dat.load_data('../../data/2009_skill_builder_data_corrected/test_pid.txt')

logging.getLogger().setLevel(logging.INFO)

akt = AKT(n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size, maxgradnorm)
akt.train(train_data, test_data, epoch=2)
akt.save("akt.params")

akt.load("akt.params")
_, auc, accuracy = akt.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
