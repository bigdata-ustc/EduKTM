import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .modules import UKSE, KSE, OTE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OKTNet(nn.Module):
    def __init__(self, n_skill, n_exercise, n_it, n_at, d_e, d_q, d_a, d_at, d_p, d_h, dropout=0.05):
        super(OKTNet, self).__init__()
        self.device = device

        self.n_skill = n_skill
        self.n_at = n_at
        self.d_h = d_h
        self.d_a = d_a

        d_it = d_h
        self.it_embed = nn.Embedding(n_it + 1, d_it)
        xavier_uniform_(self.it_embed.weight)
        self.at_embed = nn.Embedding(n_at + 1, d_at)
        xavier_uniform_(self.at_embed.weight)
        self.answer_embed = nn.Embedding(2, d_a)
        xavier_uniform_(self.answer_embed.weight)
        self.exercise_embed = nn.Embedding(n_exercise + 1, d_e)
        xavier_uniform_(self.exercise_embed.weight)
        self.skill_embed = nn.Embedding(n_skill + 1, d_q)
        xavier_uniform_(self.skill_embed.weight)

        self.linear_q = nn.Linear(d_e + d_q, d_p)
        xavier_uniform_(self.linear_q.weight)
        if n_at == 0:
            self.linear_x = nn.Linear(d_p + d_a, d_h)
        else:
            self.linear_x = nn.Linear(d_at + d_p + d_a, d_h)
        xavier_uniform_(self.linear_x.weight)
        self.ukse = UKSE(d_h)
        self.kse = KSE(d_h)
        self.ote = OTE(d_it, d_h, d_h)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(dropout)

        self.predict = nn.Sequential(
            nn.Linear(d_h + d_p, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, kc_data, a_data, e_data, it_data, at_data):
        # prepare data
        batch_size, seq_len = kc_data.size(0), kc_data.size(1)

        E = self.exercise_embed(e_data)
        KC = self.skill_embed(kc_data)
        IT = self.it_embed(it_data)
        Ans = self.answer_embed(a_data)
        Q = self.linear_q(torch.cat((E, KC), 2))
        if self.n_at == 0:
            X = self.linear_x(torch.cat((Q, Ans), 2))
        else:
            AT = self.at_embed(at_data)
            X = self.linear_x(torch.cat((Q, Ans, AT), 2))

        previous_h = xavier_uniform_(torch.zeros(1, self.d_h)).repeat(batch_size, 1).to(self.device)
        v = xavier_uniform_(torch.empty(1, self.d_h)).repeat(batch_size, 1).to(self.device)
        pred = torch.zeros(batch_size, seq_len, 1).to(self.device)

        for t in range(seq_len):
            it_embed = IT[:, t]
            q = Q[:, t]
            x = X[:, t]

            # predict
            updated_h = self.ukse(previous_h, v, it_embed)
            pred[:, t] = self.sig(self.predict(torch.cat((updated_h, q), 1)))

            # update
            h = self.kse(x, updated_h)
            v = self.ote(previous_h, h, it_embed, v)

            previous_h = h

        return pred.squeeze(-1)
