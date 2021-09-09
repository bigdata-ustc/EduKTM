# coding: utf-8
# 2021/5/24 @ tongshiwei

import dgl
import torch
from EduKTM import KTM
from EduKTM.utils import SLMLoss
from tqdm import tqdm
import numpy as np
from .net import GKTNet


class GKT(KTM):
    def __init__(self, graph: dgl.DGLGraph, item_dim, state_dim):
        super(GKT, self).__init__()
        self.net = GKTNet(graph, graph.num_nodes(), item_dim, state_dim)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        loss_function = SLMLoss()

        trainer = torch.optim.Adam(self.net.parameters(), lr)

        self.net = self.net.to(device)
        for e in range(epoch):
            losses = []
            for (item_id, response, data_mask, label, next_item_id, label_mask) in tqdm(train_data, "Epoch %s" % e):
                # convert to device
                item_id: torch.Tensor = item_id.to(device)
                response: torch.Tensor = response.to(device)
                data_mask: torch.Tensor = data_mask.to(device)
                label: torch.Tensor = label.to(device)
                next_item_id: torch.Tensor = next_item_id.to(device)
                label_mask: torch.Tensor = label_mask.to(device)

                # real training
                predicted_response, _ = self.net(item_id, response, data_mask)
                loss = loss_function(predicted_response, next_item_id, label, label_mask)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] SLMoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, *args, **kwargs) -> ...:
        pass

    def save(self, *args, **kwargs) -> ...:
        pass

    def load(self, *args, **kwargs) -> ...:
        pass
