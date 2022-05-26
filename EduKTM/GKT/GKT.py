# coding: utf-8
# 2022/2/25 @ fannazya

import logging
import numpy as np
import torch
from tqdm import tqdm
from EduKTM import KTM
from .GKTNet import GKTNet
from EduKTM.utils import SLMLoss, tensor2list, pick
from sklearn.metrics import roc_auc_score, accuracy_score


class GKT(KTM):
    def __init__(self, ku_num, graph, hidden_num, net_params: dict = None, loss_params=None):
        super(GKT, self).__init__()
        self.gkt_model = GKTNet(
            ku_num,
            graph,
            hidden_num,
            **(net_params if net_params is not None else {})
        )
        # self.gkt_model = GKTNet(ku_num, graph, hidden_num)
        self.loss_params = loss_params if loss_params is not None else {}

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        loss_function = SLMLoss(**self.loss_params)
        trainer = torch.optim.Adam(self.gkt_model.parameters(), lr)

        for e in range(epoch):
            losses = []
            for (question, data, data_mask, label, pick_index, label_mask) in tqdm(train_data, "Epoch %s" % e):
                # convert to device
                question: torch.Tensor = question.to(device)
                data: torch.Tensor = data.to(device)
                data_mask: torch.Tensor = data_mask.to(device)
                label: torch.Tensor = label.to(device)
                pick_index: torch.Tensor = pick_index.to(device)
                label_mask: torch.Tensor = label_mask.to(device)

                # real training
                predicted_response, _ = self.gkt_model(question, data, data_mask)

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
        self.gkt_model.eval()
        y_true = []
        y_pred = []

        for (question, data, data_mask, label, pick_index, label_mask) in tqdm(test_data, "evaluating"):
            # convert to device
            question: torch.Tensor = question.to(device)
            data: torch.Tensor = data.to(device)
            data_mask: torch.Tensor = data_mask.to(device)
            label: torch.Tensor = label.to(device)
            pick_index: torch.Tensor = pick_index.to(device)
            label_mask: torch.Tensor = label_mask.to(device)

            # real evaluating
            output, _ = self.gkt_model(question, data, data_mask)
            output = output[:, :-1]
            output = pick(output, pick_index.to(output.device))
            pred = tensor2list(output)
            label = tensor2list(label)
            for i, length in enumerate(label_mask.numpy().tolist()):
                length = int(length)
                y_true.extend(label[i][:length])
                y_pred.extend(pred[i][:length])
        self.gkt_model.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath) -> ...:
        torch.save(self.gkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.gkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
