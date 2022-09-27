import math
import logging
import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn import metrics
from scipy.stats import pearsonr

from EduKTM import KTM
from .OKTNet import OKTNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred = all_pred.copy()
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def compute_rmse(all_target, all_pred):
    return np.sqrt(metrics.mean_squared_error(all_target, all_pred))


def compute_r2(all_target, all_pred):
    return np.power(pearsonr(all_target, all_pred)[0], 2)


def train_one_epoch(net, optimizer, criterion, batch_size, q_data, a_data, e_data, it_data, at_data=None):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[shuffled_ind]
    e_data = e_data[shuffled_ind]
    if at_data is not None:
        at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]

    pred_list = []
    target_list = []
    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()

        q_one_seq = q_data[idx * batch_size: (idx + 1) * batch_size, :]
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

        input_q = torch.from_numpy(q_one_seq).long().to(device)
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_a = torch.from_numpy(a_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        input_at = None
        if at_data is not None:
            at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
            input_at = torch.from_numpy(at_one_seq).long().to(device)

        pred = net(input_q, input_a, input_e, input_it, input_at)

        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth)

        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        pred_list.append(masked_pred)
        target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    r2 = compute_r2(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, r2, auc, accuracy


def test_one_epoch(net, batch_size, q_data, a_data, e_data, it_data, at_data=None):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []
    mask_list = []

    for idx in tqdm.tqdm(range(n), 'Testing'):
        q_one_seq = q_data[idx * batch_size: (idx + 1) * batch_size, :]
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

        input_q = torch.from_numpy(q_one_seq).long().to(device)
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_a = torch.from_numpy(a_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        input_at = None
        if at_data is not None:
            at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
            input_at = torch.from_numpy(at_one_seq).long().to(device)

        with torch.no_grad():
            pred = net(input_q, input_a, input_e, input_it, input_at)

            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)
            mask_list.append(mask.long().cpu().numpy())

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    mask_list = np.concatenate(mask_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    r2 = compute_r2(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    rmse = compute_rmse(all_target, all_pred)

    return loss, rmse, r2, auc, accuracy


class OKT(KTM):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_e, d_q, d_a, d_at, d_p, d_h, batch_size=64, dropout=0.2):
        super(OKT, self).__init__()

        self.okt_net = OKTNet(n_question, n_exercise, n_it, n_at, d_e, d_q, d_a, d_at, d_p, d_h,
                              dropout=dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5,
              filepath=None) -> ...:
        optimizer = torch.optim.Adam(self.okt_net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss()
        best_train_auc, best_test_auc = .0, .0

        for idx in range(epoch):
            train_loss, train_r2, train_auc, train_accuracy = train_one_epoch(self.okt_net, optimizer, criterion,
                                                                              self.batch_size, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            if train_auc > best_train_auc:
                best_train_auc = train_auc

            if test_data is not None:
                _, _, test_r2, test_auc, test_accuracy = self.eval(test_data)
                print("[Epoch %d] r2: %.6f, auc: %.6f, accuracy: %.6f" % (idx, test_r2, test_auc, test_accuracy))
                scheduler.step()
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    if filepath is not None:
                        self.save(filepath)

        return best_train_auc, best_test_auc

    def eval(self, test_data) -> ...:
        return test_one_epoch(self.okt_net, self.batch_size, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.okt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.okt_net.load_state_dict(torch.load(filepath, map_location='cpu'))
        logging.info("load parameters from %s" % filepath)
