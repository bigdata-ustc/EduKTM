# coding: utf-8
# 2023/11/21 @ xubihan

from sklearn import metrics
from sklearn.metrics import mean_squared_error
import logging
import torch
import torch.nn as nn
import numpy as np
from .model import Recurrent
from EduKTM import KTM
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) \
        + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def train_one_epoch(recurrent, optimizer, criterion,
                    batch_size, Topics_all, Resps_all,
                    time_factor_all, attempts_factor_all, hints_factor_all):
    recurrent.train()
    all_pred = []
    all_target = []
    n = len(Topics_all) // batch_size
    shuffled_ind = np.arange(len(Topics_all))
    np.random.shuffle(shuffled_ind)
    Topics_all = Topics_all[shuffled_ind]
    Resps_all = Resps_all[shuffled_ind]
    time_factor_all = time_factor_all[shuffled_ind]
    attempts_factor_all = attempts_factor_all[shuffled_ind]
    hints_factor_all = hints_factor_all[shuffled_ind]

    for idx in tqdm(range(n)):
        optimizer.zero_grad()

        Topics = Topics_all[idx * batch_size: (idx + 1) * batch_size, :]
        Resps = Resps_all[idx * batch_size: (idx + 1) * batch_size, :]
        time_factor = time_factor_all[idx * batch_size:
                                      (idx + 1) * batch_size, :]
        attempts_factor = attempts_factor_all[idx * batch_size:
                                              (idx + 1) * batch_size, :]
        hints_factor = hints_factor_all[idx * batch_size:
                                        (idx + 1) * batch_size, :]

        input_topics = torch.from_numpy(Topics).long().to(device)
        input_resps = torch.from_numpy(Resps).long().to(device)
        input_time_factor = torch.from_numpy(time_factor).float().to(device)
        input_attempts_factor = torch.from_numpy(
            attempts_factor).float().to(device)
        input_hints_factor = torch.from_numpy(hints_factor).float().to(device)

        y_pred = recurrent(input_topics, input_resps, input_time_factor,
                           input_attempts_factor, input_hints_factor)

        mask = input_topics[:, 1:] > 0
        masked_pred = y_pred[:, 1:][mask]
        masked_truth = input_resps[:, 1:][mask]
        loss = criterion(masked_pred, masked_truth.float()).sum()
        loss.backward()
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        all_pred.append(masked_pred)
        all_target.append(masked_truth)

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)

    return loss, auc, acc


def test_one_epoch(recurrent, batch_size, Topics_all, Resps_all,
                   time_factor_all, attempts_factor_all, hints_factor_all):
    recurrent.eval()
    all_pred, all_target = [], []
    n = len(Topics_all) // batch_size
    for idx in range(n):
        Topics = Topics_all[idx * batch_size:
                            (idx + 1) * batch_size, :]
        Resps = Resps_all[idx * batch_size:
                          (idx + 1) * batch_size, :]
        time_factor = time_factor_all[idx * batch_size:
                                      (idx + 1) * batch_size, :]
        attempts_factor = attempts_factor_all[idx * batch_size:
                                              (idx + 1) * batch_size, :]
        hints_factor = hints_factor_all[idx * batch_size:
                                        (idx + 1) * batch_size, :]

        input_topics = torch.from_numpy(Topics).long().to(device)
        input_resps = torch.from_numpy(Resps).long().to(device)
        input_time_factor = torch.from_numpy(time_factor).float().to(device)
        input_attempts_factor = torch.from_numpy(attempts_factor)\
            .float().to(device)
        input_hints_factor = torch.from_numpy(hints_factor)\
            .float().to(device)

        with torch.no_grad():
            y_pred = recurrent(input_topics, input_resps, input_time_factor,
                               input_attempts_factor, input_hints_factor)

            mask = input_topics[:, 1:] > 0
            masked_pred = y_pred[:, 1:][mask]
            masked_truth = input_resps[:, 1:][mask]

            masked_pred = masked_pred.detach().cpu().numpy()
            masked_truth = masked_truth.detach().cpu().numpy()

            all_pred.append(masked_pred)
            all_target.append(masked_truth)

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    rmse = mean_squared_error(all_target, all_pred, squared=False)
    acc = compute_accuracy(all_target, all_pred)

    return loss, auc, acc, rmse


class LBKT(KTM):
    def __init__(self, num_topics, dim_tp, num_resps, num_units,
                 dropout, dim_hidden, memory_size, BATCH_SIZE, q_matrix):
        super(LBKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.recurrent = Recurrent(num_topics, dim_tp, num_resps, num_units,
                                   dropout, dim_hidden, memory_size,
                                   BATCH_SIZE, q_matrix).to(device)
        self.batch_size = BATCH_SIZE

    def train(self, train_data, test_data, epoch: int,
              lr, lr_decay_step=1, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.recurrent.parameters(), lr=lr,
                                     eps=1e-8, betas=(0.1, 0.999),
                                     weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')

        best_test_auc = 0
        for idx in range(epoch):
            train_loss, _, _ = train_one_epoch(self.recurrent,
                                               optimizer, criterion,
                                               self.batch_size, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            scheduler.step()
            if test_data is not None:
                _, valid_auc, valid_acc, valid_rmse = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (
                    idx, valid_auc, valid_acc, valid_rmse))
                if valid_auc > best_test_auc:
                    best_test_auc = valid_auc
        return best_test_auc

    def eval(self, test_data) -> ...:
        self.recurrent.eval()
        return test_one_epoch(self.recurrent, self.batch_size, *test_data)

    def save(self, filepath) -> ...:

        torch.save(self.recurrent.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.recurrent.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
