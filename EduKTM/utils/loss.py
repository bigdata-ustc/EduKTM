# coding: utf-8
# 2021/5/24 @ tongshiwei
__all__ = ["SequenceLogisticMaskLoss", "LogisticMaskLoss"]

import torch
from torch import nn

from .torch import pick, sequence_mask


class SequenceLogisticMaskLoss(nn.Module):
    """
    Notes
    -----
    The loss has been average, so when call the step method of trainer, batch_size should be 1
    """

    def __init__(self, lr=0.0, lw1=0.0, lw2=0.0):
        """

        Parameters
        ----------
        lr: reconstruction
        lw1
        lw2
        """
        super(SequenceLogisticMaskLoss, self).__init__()
        self.lr = lr
        self.lw1 = lw1
        self.lw2 = lw2
        self.loss = torch.nn.BCELoss(reduction='none')

    def forward(self, pred_rs, pick_index, label, label_mask):
        if self.lw1 > 0.0 or self.lw2 > 0.0:
            post_pred_rs = pred_rs[:, 1:]
            pre_pred_rs = pred_rs[:, :-1]
            diff = post_pred_rs - pre_pred_rs
            diff = sequence_mask(diff, label_mask)
            w1 = torch.mean(torch.norm(diff, 1, -1)) / diff.shape[-1]
            w2 = torch.mean(torch.norm(diff, 2, -1)) / diff.shape[-1]
            # w2 = F.mean(F.sqrt(diff ** 2))
            w1 = w1 * self.lw1 if self.lw1 > 0.0 else 0.0
            w2 = w2 * self.lw2 if self.lw2 > 0.0 else 0.0
        else:
            w1 = 0.0
            w2 = 0.0

        if self.lr > 0.0:
            re_pred_rs = pred_rs[:, 1:]
            re_pred_rs = pick(re_pred_rs, pick_index)
            wr = sequence_mask(self.loss(re_pred_rs, label.float()), label_mask)
            wr = torch.mean(wr) * self.lr
        else:
            wr = 0.0

        pred_rs = pred_rs[:, 1:]
        pred_rs = pick(pred_rs, pick_index)
        loss = sequence_mask(self.loss(pred_rs, label.float()), label_mask)
        # loss = F.sum(loss, axis=-1)
        loss = torch.mean(loss) + w1 + w2 + wr
        return loss


class LogisticMaskLoss(nn.Module):  # pragma: no cover
    """
    Notes
    -----
    The loss has been average, so when call the step method of trainer, batch_size should be 1
    """

    def __init__(self):
        super(LogisticMaskLoss, self).__init__()

        self.loss = torch.nn.BCELoss()

    def forward(self, pred_rs, label, label_mask, *args, **kwargs):
        loss = sequence_mask(self.loss(pred_rs, label), label_mask)
        loss = torch.mean(loss)
        return loss
