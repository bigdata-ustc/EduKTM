# coding: utf-8
# 2021/4/23 @ zengxiaonan

import logging

import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from EduKTM import KTM


class Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.RNN(num_questions * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, _ = self.rnn(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    return pred, truth


class DKT(KTM):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = Net(num_questions, hidden_size, num_layers)

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                integrated_pred = self.dkt_model(batch)
                batch_size = batch.shape[0]
                loss = torch.Tensor([0.0])
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                    if pred.shape[0] != 0:
                        loss += loss_function(pred, truth.float())

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))

    def eval(self, test_data) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        for batch in tqdm.tqdm(test_data, "evaluating"):
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])

        return roc_auc_score(y_truth.detach().numpy(), y_pred.detach().numpy())

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
