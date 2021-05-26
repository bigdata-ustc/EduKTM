# coding: utf-8
# 2021/5/25 @ tongshiwei

import logging
import torch
from EduKTM import KTM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from EduKTM.utils import sequence_mask, SLMLoss, tensor2list, pick
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


class DKTNet(nn.Module):
    def __init__(self, ku_num, hidden_num, add_embedding_layer=False, embedding_dim=None, dropout=0.0, **kwargs):
        super(DKTNet, self).__init__()
        self.ku_num = ku_num
        self.hidden_dim = hidden_num
        self.output_dim = ku_num
        if add_embedding_layer is True:
            embedding_dim = self.hidden_dim if embedding_dim is None else embedding_dim
            self.embeddings = nn.Sequential(
                nn.Embedding(ku_num * 2, embedding_dim),
                nn.Dropout(kwargs.get("embedding_dropout", 0.2))
            )
            rnn_input_dim = embedding_dim
        else:
            self.embeddings = lambda x: F.one_hot(x, num_classes=self.output_dim * 2).float()
            rnn_input_dim = ku_num * 2

        self.rnn = nn.RNN(rnn_input_dim, hidden_num, 1, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, responses, mask=None, begin_state=None):
        responses = self.embeddings(responses)
        output, hn = self.rnn(responses)
        output = self.sig(self.fc(self.dropout(output)))
        if mask is not None:
            output = sequence_mask(output, mask)
        return output, hn


class DKTPlus(KTM):
    def __init__(self, ku_num, hidden_num, net_params: dict = None, loss_params=None):
        super(DKTPlus, self).__init__()
        self.dkt_net = DKTNet(
            ku_num,
            hidden_num,
            **(net_params if net_params is not None else {})
        )
        self.loss_params = loss_params if loss_params is not None else {}

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        loss_function = SLMLoss(**self.loss_params)

        trainer = torch.optim.Adam(self.dkt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for (data, data_mask, label, pick_index, label_mask) in tqdm(train_data, "Epoch %s" % e):
                # convert to device
                data: torch.Tensor = data.to(device)
                data_mask: torch.Tensor = data_mask.to(device)
                label: torch.Tensor = label.to(device)
                pick_index: torch.Tensor = pick_index.to(device)
                label_mask: torch.Tensor = label_mask.to(device)

                # real training
                predicted_response, _ = self.dkt_net(data, data_mask)
                loss = loss_function(predicted_response, pick_index, label, label_mask)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] SLMoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.dkt_net.eval()
        y_true = []
        y_pred = []

        for (data, data_mask, label, pick_index, label_mask) in tqdm(test_data, "evaluating"):
            # convert to device
            data: torch.Tensor = data.to(device)
            data_mask: torch.Tensor = data_mask.to(device)
            label: torch.Tensor = label.to(device)
            pick_index: torch.Tensor = pick_index.to(device)
            label_mask: torch.Tensor = label_mask.to(device)

            # real evaluating
            output, _ = self.dkt_net(data, data_mask)
            output = output[:, :-1]
            output = pick(output, pick_index.to(output.device))
            pred = tensor2list(output)
            label = tensor2list(label)
            for i, length in enumerate(label_mask.numpy().tolist()):
                length = int(length)
                y_true.extend(label[i][:length])
                y_pred.extend(pred[i][:length])
        self.dkt_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath) -> ...:
        torch.save(self.dkt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
