# coding: utf-8
# 2022/3/1 @ fannazya
__all__ = ["GKTNet"]

import json
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
from EduKTM.utils import GRUCell, begin_states, get_states, expand_tensor, \
    format_sequence, mask_sequence_variable_length


class GKTNet(nn.Module):
    def __init__(self, ku_num, graph, hidden_num=None, latent_dim=None, dropout=0.0):
        super(GKTNet, self).__init__()
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.ku_num if latent_dim is None else int(latent_dim)
        self.neighbor_dim = self.hidden_num + self.latent_dim
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(list(range(ku_num)))
        try:
            with open(graph) as f:
                self.graph.add_weighted_edges_from(json.load(f))
        except ValueError:
            with open(graph) as f:
                self.graph.add_weighted_edges_from([e + [1.0] for e in json.load(f)])

        self.rnn = GRUCell(self.hidden_num)
        self.response_embedding = nn.Embedding(2 * self.ku_num, self.latent_dim)
        self.concept_embedding = nn.Embedding(self.ku_num, self.latent_dim)
        self.f_self = nn.Linear(self.neighbor_dim, self.hidden_num)
        self.n_out = nn.Linear(2 * self.neighbor_dim, self.hidden_num)
        self.n_in = nn.Linear(2 * self.neighbor_dim, self.hidden_num)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_num, 1)

    def in_weight(self, x, ordinal=True, with_weight=True):
        if isinstance(x, torch.Tensor):
            x = x.numpy().tolist()
        if isinstance(x, list):
            return [self.in_weight(_x) for _x in x]
        elif isinstance(x, (int, float)):
            _ret = [0] * self.ku_num
            for i in self.graph.predecessors(int(x)):
                _ret[i] = 1
            return _ret

    def out_weight(self, x, ordinal=True, with_weight=True):
        if isinstance(x, torch.Tensor):
            x = x.numpy().tolist()
        if isinstance(x, list):
            return [self.out_weight(_x) for _x in x]
        elif isinstance(x, (int, float)):
            _ret = [0] * self.ku_num
            for i in self.graph.successors(int(x)):
                _ret[i] = 1
            return _ret

    def neighbors(self, x, ordinal=True, with_weight=False):
        if isinstance(x, torch.Tensor):
            x = x.numpy().tolist()
        if isinstance(x, list):
            return [self.neighbors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            _ret = [0] * self.ku_num
            for i in self.graph.neighbors(int(x)):
                _ret[i] = 1
            return _ret

    def forward(self, questions, answers, valid_length=None, compressed_out=True, layout="NTC"):
        length = questions.shape[1]
        inputs, axis, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _ = format_sequence(length, answers, layout, False)

        states = begin_states([(batch_size, self.ku_num, self.hidden_num)])[0]
        outputs = []
        for i in range(length):
            # neighbors - aggregate
            inputs_i = inputs[i].reshape([batch_size, ])
            answer_i = answers[i].reshape([batch_size, ])

            _neighbors = self.neighbors(inputs_i)
            neighbors_mask = expand_tensor(torch.Tensor(_neighbors), -1, self.hidden_num)
            _neighbors_mask = expand_tensor(torch.Tensor(_neighbors), -1, self.hidden_num + self.latent_dim)

            # get concept embedding
            concept_embeddings = self.concept_embedding.weight.data
            concept_embeddings = expand_tensor(concept_embeddings, 0, batch_size)

            agg_states = torch.cat((concept_embeddings, states), dim=-1)

            # aggregate
            _neighbors_states = _neighbors_mask * agg_states

            # self - aggregate
            _concept_embedding = get_states(inputs_i, states)
            _self_hidden_states = torch.cat((_concept_embedding, self.response_embedding(answer_i)), dim=-1)

            _self_mask = F.one_hot(inputs_i, self.ku_num)  # p
            _self_mask = expand_tensor(_self_mask, -1, self.hidden_num)

            self_hidden_states = expand_tensor(_self_hidden_states, 1, self.ku_num)

            # aggregate
            _hidden_states = torch.cat((_neighbors_states, self_hidden_states), dim=-1)

            _in_state = self.n_in(_hidden_states)
            _out_state = self.n_out(_hidden_states)
            in_weight = expand_tensor(torch.Tensor(self.in_weight(inputs_i)), -1, self.hidden_num)
            out_weight = expand_tensor(torch.Tensor(self.out_weight(inputs_i)), -1, self.hidden_num)

            next_neighbors_states = in_weight * _in_state + out_weight * _out_state

            # self - update
            next_self_states = self.f_self(_self_hidden_states)
            next_self_states = expand_tensor(next_self_states, 1, self.ku_num)
            next_self_states = _self_mask * next_self_states

            next_states = neighbors_mask * next_neighbors_states + next_self_states

            next_states, _ = self.rnn(next_states, [states])
            next_states = (_self_mask + neighbors_mask) * next_states + (1 - _self_mask - neighbors_mask) * states

            states = self.dropout(next_states)
            output = torch.sigmoid(self.out(states).squeeze(axis=-1))  # p
            outputs.append(output)

        if valid_length is not None:
            if compressed_out:
                states = None
            outputs = mask_sequence_variable_length(torch, outputs, valid_length)

        return outputs, states
