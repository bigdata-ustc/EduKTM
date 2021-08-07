# coding: utf-8
# 2021/7/15 @ sone

import logging
import math
import torch
import numpy as np
from sklearn import metrics

from EduKTM import KTM
from .AKTNet import AKTNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, params, optimizer, q_data, qa_data, pid_data):
    net.train()
    pid_flag, batch_size, n_question, maxgradnorm = (
        params['is_pid'], params['batch_size'], params['n_question'], params['maxgradnorm'])
    n = int(math.ceil(len(q_data) / batch_size))
    q_data = q_data.T
    qa_data = qa_data.T
    # shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    if pid_flag:
        pid_data = pid_data.T
        pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    true_el = 0
    for idx in range(n):
        optimizer.zero_grad()

        q_one_seq = q_data[:, idx * batch_size: (idx + 1) * batch_size]
        qa_one_seq = qa_data[:, idx * batch_size: (idx + 1) * batch_size]

        input_q = np.transpose(q_one_seq[:, :])
        input_qa = np.transpose(qa_one_seq[:, :])
        target = np.transpose(qa_one_seq[:, :])
        target = (target - 1) / n_question
        target_1 = np.floor(target)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            pid_one_seq = pid_data[:, idx * batch_size: (idx + 1) * batch_size]
            input_pid = np.transpose(pid_one_seq[:, :])
            input_pid = torch.from_numpy(input_pid).long().to(device)

            loss, pred, true_ct = net(input_q, input_qa, target, input_pid)
        else:
            loss, pred, true_ct = net(input_q, input_qa, target)
        pred = pred.detach().cpu().numpy()
        loss.backward()
        true_el += true_ct.cpu().numpy()

        if maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


def test_one_epoch(net, params, q_data, qa_data, pid_data):
    pid_flag, batch_size, n_question = params['is_pid'], params['batch_size'], params['n_question']
    net.eval()
    n = int(math.ceil(len(q_data) / batch_size))
    q_data = q_data.T
    qa_data = qa_data.T
    if pid_flag:
        pid_data = pid_data.T
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0

    for idx in range(n):
        q_one_seq = q_data[:, idx * batch_size: (idx + 1) * batch_size]
        qa_one_seq = qa_data[:, idx * batch_size: (idx + 1) * batch_size]

        input_q = np.transpose(q_one_seq[:, :])
        input_qa = np.transpose(qa_one_seq[:, :])
        target = np.transpose(qa_one_seq[:, :])
        target = (target - 1) / n_question
        target_1 = np.floor(target)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            pid_one_seq = pid_data[:, idx * batch_size: (idx + 1) * batch_size]
            input_pid = np.transpose(pid_one_seq[:, :])
            input_pid = torch.from_numpy(input_pid).long().to(device)

        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = net(input_q, input_qa, target, input_pid)
            else:
                loss, pred, ct = net(input_q, input_qa, target)
        pred = pred.cpu().numpy()
        true_el += ct.cpu().numpy()
        if (idx + 1) * batch_size > seq_num:
            real_batch_size = seq_num - idx * batch_size
            count += real_batch_size
        else:
            count += batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, 'Seq not matching'

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


class AKT(KTM):
    def __init__(self, n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size, maxgradnorm,
                 separate_qa=False):
        super(AKT, self).__init__()
        self.params = {
            'is_pid': n_pid > 0,
            'batch_size': batch_size,
            'n_question': n_question,
            'maxgradnorm': maxgradnorm,
        }
        self.akt_net = AKTNet(n_question=n_question, n_pid=n_pid, n_blocks=n_blocks, d_model=d_model, dropout=dropout,
                              kq_same=kq_same, l2=l2, separate_qa=separate_qa).to(device)

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        optimizer = torch.optim.Adam(self.akt_net.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)

        for idx in range(epoch):
            train_loss, train_accuracy, train_acc = train_one_epoch(self.akt_net, self.params, optimizer, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))

            if test_data is not None:
                valid_loss, valid_accuracy, valid_acc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, valid_acc, valid_accuracy))

    def eval(self, test_data) -> ...:
        self.akt_net.eval()
        return test_one_epoch(self.akt_net, self.params, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.akt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.akt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
