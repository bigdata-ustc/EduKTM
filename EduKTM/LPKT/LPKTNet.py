# coding: utf-8
# 2021/8/17 @ sone

import torch
from torch import nn


class LPKTNet(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout=0.2):
        super(LPKTNet, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.q_matrix = q_matrix
        self.n_question = n_question

        self.at_embed = nn.Embedding(n_at + 1, d_k)
        self.it_embed = nn.Embedding(n_it + 1, d_k)
        self.e_embed = nn.Embedding(n_exercise + 1, d_e)
        self.h_embed = nn.Embedding(n_question + 1, d_k)

        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_data, at_data, a_data, it_data):
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        h_pre = torch.zeros(batch_size, self.n_question + 1, self.d_k)
        h_tilde_pre = None
        learning_pre = torch.zeros(batch_size, self.d_k)

        pred = torch.zeros(batch_size, seq_len)

        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            q_e = self.q_matrix[e].view(batch_size, 1, -1)
            e_embed = e_embed_data[:, t]
            at = at_embed_data[:, t]
            a = a_data[:, t]
            it = it_embed_data[:, t]

            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            learning = self.linear_1(torch.cat((e_embed, at, a), 1))
            learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(LG.view(batch_size, self.d_k, 1).bmm(q_e).transpose(1, 2))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            # W_4: (3 * d_k, d_k)
            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            y = self.linear_5(torch.cat((e_embed, h_tilde), 1)).sum(1) / self.d_k
            y = self.sig(y)
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred
