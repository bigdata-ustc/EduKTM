import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class OTE(nn.Module):
    def __init__(self, d_h, d_it, d_v):
        super(OTE, self).__init__()
        self.linear_v = nn.Linear(2 * d_h + d_it, d_v)
        self.linear_p = nn.Linear(d_it + d_v, d_v)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, previous_h, h, it, v):
        delta = torch.cat((previous_h, h), 1)
        v_prime = self.tanh(self.linear_v(torch.cat((delta, it), 1)))
        p = self.sig(self.linear_p(torch.cat((v, it), 1)))
        v = (1 - p) * v + p * v_prime
        return v


class UKSE(nn.Module):
    def __init__(self, d_h):
        super(UKSE, self).__init__()
        self.linear_h = nn.Linear(3 * d_h, d_h)
        self.linear_p = nn.Linear(3 * d_h, d_h)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h, v, it):
        h_prime = self.tanh(self.linear_h(torch.cat((h, v, it), 1)))
        p = self.sig(self.linear_p(torch.cat((h, v, it), 1)))
        return (1 - p) * h + p * h_prime


class KSE(nn.Module):
    def __init__(self, d_h):
        super(KSE, self).__init__()
        self.linear_h = nn.Linear(2 * d_h, d_h)
        self.linear_p = nn.Linear(2 * d_h, d_h)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hr):
        h_tilde = self.tanh(self.linear_h(torch.cat((x, hr), 1)))
        p = self.sig(self.linear_p(torch.cat((x, hr), 1)))
        hx = (1 - p) * hr + p * h_tilde
        return hx
