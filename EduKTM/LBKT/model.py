# coding: utf-8
# 2023/11/21 @ xubihan

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Layer1(nn.Module):
    def __init__(self, num_units, d=10, k=0.3, b=0.3, name='lb'):
        super(Layer1, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(2 * num_units, num_units))
        self.bias = nn.Parameter(torch.zeros(1, num_units))

        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.bias)

        self.d = d
        self.k = k
        self.b = b

    def forward(self, factor, interact_emb, h):
        k = self.k
        d = self.d
        b = self.b

        gate = k + (1 - k) / (1 + torch.exp(-d * (factor - b)))

        w = torch.cat([h, interact_emb], -1).matmul(self.weight) + self.bias

        w = nn.Sigmoid()(w * gate)
        return w


class LBKTcell(nn.Module):
    def __init__(self, num_units, memory_size, dim_tp,
                 dropout=0.2, name='lbktcell'):
        super(LBKTcell, self).__init__()
        self.num_units = num_units
        self.memory_size = memory_size
        self.dim_tp = dim_tp
        self.r = 4
        self.factor_dim = 50

        self.time_gain = Layer1(self.num_units, name='time_gain')
        self.attempt_gain = Layer1(self.num_units, name='attempt_gain')
        self.hint_gain = Layer1(self.num_units, name='hint_gain')

        self.time_weight = nn.Parameter(torch.Tensor(self.r, num_units + 1, num_units))
        nn.init.xavier_normal_(self.time_weight)

        self.attempt_weight = nn.Parameter(torch.Tensor(self.r, num_units + 1, num_units))
        nn.init.xavier_normal_(self.attempt_weight)

        self.hint_weight = nn.Parameter(torch.Tensor(self.r, num_units + 1, num_units))
        nn.init.xavier_normal_(self.hint_weight)

        self.Wf = nn.Parameter(torch.Tensor(1, self.r))
        nn.init.xavier_normal_(self.Wf)

        self.bias = nn.Parameter(torch.Tensor(1, num_units))
        nn.init.xavier_normal_(self.bias)

        self.gate3 = nn.Linear(2 * num_units + 3 * self.factor_dim, num_units)
        torch.nn.init.xavier_normal_(self.gate3.weight)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_tp + num_units, num_units)
        torch.nn.init.xavier_normal_(self.output_layer.weight)
        self.sig = nn.Sigmoid()

    def forward(self, interact_emb, correlation_weight, topic_emb,
                time_factor, attempt_factor, hint_factor, h_pre):
        # bs *1 * memory_size , bs * memory_size * d_k
        h_pre_tilde = torch.squeeze(torch.bmm(correlation_weight.unsqueeze(1), h_pre), 1)
        # predict performance
        preds = torch.sum(self.sig(self.output_layer(torch.cat([h_pre_tilde, topic_emb], -1))),
                          -1) / self.num_units  # bs

        # characterize each behavior's effect
        time_gain = self.time_gain(time_factor, interact_emb, h_pre_tilde)
        attempt_gain = self.attempt_gain(attempt_factor, interact_emb, h_pre_tilde)
        hint_gain = self.hint_gain(hint_factor, interact_emb, h_pre_tilde)

        # capture the dependency among different behaviors
        pad = torch.ones_like(time_factor)  # bs * 1
        time_gain1 = torch.cat([time_gain, pad], -1)  # bs * num_units + 1
        attempt_gain1 = torch.cat([attempt_gain, pad], -1)
        hint_gain1 = torch.cat([hint_gain, pad], -1)
        # bs * r  *num_units: bs * num_units + 1 ,r * num_units + 1 *num_units
        fusion_time = torch.matmul(time_gain1, self.time_weight)
        fusion_attempt = torch.matmul(attempt_gain1, self.attempt_weight)
        fusion_hint = torch.matmul(hint_gain1, self.hint_weight)
        fusion_all = fusion_time * fusion_attempt * fusion_hint
        # 1 * r, bs * r * num_units -> bs * 1 * num_units -> bs * num_units
        fusion_all = torch.matmul(self.Wf, fusion_all.permute(1, 0, 2)).squeeze(1) + self.bias
        learning_gain = torch.relu(fusion_all)

        LG = torch.matmul(correlation_weight.unsqueeze(-1), learning_gain.unsqueeze(1))

        # forget effect
        forget_gate = self.gate3(torch.cat([h_pre, interact_emb.unsqueeze(1).repeat(1, self.memory_size, 1),
                                            time_factor.unsqueeze(1).repeat(1, self.memory_size, self.factor_dim),
                                            attempt_factor.unsqueeze(1).repeat(1, self.memory_size, self.factor_dim),
                                            hint_factor.unsqueeze(1).repeat(1, self.memory_size, self.factor_dim)], -1))
        LG = self.dropout(LG)
        h = h_pre * self.sig(forget_gate) + LG

        return preds, h


class Recurrent(nn.Module):
    def __init__(self, num_topics, dim_tp, num_resps, num_units, dropout,
                 dim_hidden, memory_size, batch_size, q_matrix):
        super(Recurrent, self).__init__()

        self.embedding_topic = nn.Embedding(num_topics + 10, dim_tp)
        torch.nn.init.xavier_normal_(self.embedding_topic.weight)

        self.embedding_resps = nn.Embedding(num_resps, dim_hidden)
        torch.nn.init.xavier_normal_(self.embedding_resps.weight)

        self.memory_size = memory_size
        self.num_units = num_units
        self.dim_tp = dim_tp
        self.q_matrix = q_matrix

        self.input_layer = nn.Linear(dim_tp + dim_hidden, num_units)
        torch.nn.init.xavier_normal_(self.input_layer.weight)

        self.lbkt_cell = LBKTcell(num_units, memory_size,
                                  dim_tp, dropout=dropout, name='lbkt')

        self.init_h = nn.Parameter(torch.Tensor(memory_size, num_units))
        nn.init.xavier_normal_(self.init_h)

    def forward(self, topics, resps, time_factor, attempt_factor, hint_factor):
        batch_size, seq_len = topics.size(0), topics.size(1)
        topic_emb = self.embedding_topic(topics)
        resps_emb = self.embedding_resps(resps)

        correlation_weight = self.q_matrix[topics]
        acts_emb = torch.relu(self.input_layer(torch.cat([topic_emb, resps_emb], -1)))

        time_factor = time_factor.unsqueeze(-1)
        attempt_factor = attempt_factor.unsqueeze(-1)
        hint_factor = hint_factor.unsqueeze(-1)

        h_init = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h_pre = h_init
        preds = torch.zeros(batch_size, seq_len).to(device)
        for t in range(0, seq_len):
            pred, h = self.lbkt_cell(acts_emb[:, t], correlation_weight[:, t],
                                     topic_emb[:, t], time_factor[:, t],
                                     attempt_factor[:, t], hint_factor[:, t], h_pre)
            h_pre = h

            preds[:, t] = pred

        return preds
