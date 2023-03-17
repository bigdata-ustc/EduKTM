__all__ = ["SKTNet"]


import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
from EduKTM.utils import GRUCell, begin_states, get_states, expand_tensor, \
    format_sequence, mask_sequence_variable_length
from .utils import Graph


class SKTNet(nn.Module):
    def __init__(self, ku_num, graph_params=None,
                 alpha=0.5,
                 latent_dim=None, activation=None,
                 hidden_num=90, concept_dim=None,
                 # dropout=0.5, self_dropout=0.0,
                 dropout=0.0, self_dropout=0.5,
                 # dropout=0.0, self_dropout=0.0,
                 sync_dropout=0.0,
                 prop_dropout=0.0,
                 agg_dropout=0.0,
                 params=None):
        super(SKTNet, self).__init__()
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(
            hidden_num)
        self.latent_dim = self.hidden_num if latent_dim is None else int(
            latent_dim)
        self.concept_dim = self.hidden_num if concept_dim is None else int(
            concept_dim)
        graph_params = graph_params if graph_params is not None else []
        self.graph = Graph.from_file(ku_num, graph_params)
        self.alpha = alpha

        sync_activation = nn.ReLU() if activation is None else activation
        prop_activation = nn.ReLU() if activation is None else activation
        agg_activation = nn.ReLU() if activation is None else activation

        self.rnn = GRUCell(self.hidden_num)
        self.response_embedding = nn.Embedding(
            2 * self.ku_num, self.latent_dim)
        self.concept_embedding = nn.Embedding(self.ku_num, self.concept_dim)
        self.f_self = GRUCell(self.hidden_num)
        self.self_dropout = nn.Dropout(self_dropout)
        self.f_prop = nn.Sequential(
            nn.Linear(self.hidden_num * 2, self.hidden_num),
            prop_activation,
            nn.Dropout(prop_dropout),
        )
        self.f_sync = nn.Sequential(
            nn.Linear(self.hidden_num * 3, self.hidden_num),
            sync_activation,
            nn.Dropout(sync_dropout),
        )
        self.f_agg = nn.Sequential(
            nn.Linear(self.hidden_num, self.hidden_num),
            agg_activation,
            nn.Dropout(agg_dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_num, 1)
        self.sigmoid = nn.Sigmoid()

    def neighbors(self, x, ordinal=True):
        return self.graph.neighbors(x, ordinal)

    def successors(self, x, ordinal=True):
        return self.graph.successors(x, ordinal)

    def forward(self, questions, answers, valid_length=None, states=None, layout='NTC', compressed_out=True,
                *args, **kwargs):
        length = questions.shape[1]
        device = questions.device
        inputs, axis, batch_size = format_sequence(
            length, questions, layout, False)
        answers, _, _ = format_sequence(length, answers, layout, False)
        states = begin_states([(batch_size, self.ku_num, self.hidden_num)])[0]
        states = states.to(device)
        outputs = []
        all_states = []
        for i in range(length):
            inputs_i = inputs[i].reshape([batch_size, ])
            answer_i = answers[i].reshape([batch_size, ])

            # concept embedding
            concept_embeddings = self.concept_embedding.weight.data
            concept_embeddings = expand_tensor(
                concept_embeddings, 0, batch_size)
            # concept_embeddings = (_self_mask + _successors_mask + _neighbors_mask) * concept_embeddings

            # self - influence
            _self_state = get_states(inputs_i, states)
            # fc
            # _next_self_state = self.f_self(mx.nd.concat(_self_state, self.response_embedding(answers[i]), dim=-1))
            # gru
            _next_self_state, _ = self.f_self(
                self.response_embedding(answer_i), [_self_state])
            # _next_self_state = self.f_self(mx.nd.concat(_self_hidden_states, _self_state))
            # _next_self_state, _ = self.f_self(_self_hidden_states, [_self_state])
            _next_self_state = self.self_dropout(_next_self_state)

            # get self mask
            _self_mask = torch.unsqueeze(F.one_hot(inputs_i, self.ku_num), -1)
            _self_mask = torch.broadcast_to(
                _self_mask, (-1, -1, self.hidden_num))

            # find neighbors
            _neighbors = self.neighbors(inputs_i)
            _neighbors_mask = torch.unsqueeze(
                torch.tensor(_neighbors, device=device), -1)
            _neighbors_mask = torch.broadcast_to(
                _neighbors_mask, (-1, -1, self.hidden_num))

            # synchronization
            _broadcast_next_self_states = torch.unsqueeze(_next_self_state, 1)
            _broadcast_next_self_states = torch.broadcast_to(
                _broadcast_next_self_states, (-1, self.ku_num, -1))
            # _sync_diff = mx.nd.concat(states, _broadcast_next_self_states, concept_embeddings, dim=-1)
            _sync_diff = torch.concat(
                (states, _broadcast_next_self_states, concept_embeddings), dim=-1)
            _sync_inf = _neighbors_mask * self.f_sync(_sync_diff)

            # reflection on current vertex
            _reflec_inf = torch.sum(_sync_inf, dim=1)
            _reflec_inf = torch.broadcast_to(
                torch.unsqueeze(_reflec_inf, 1), (-1, self.ku_num, -1))
            _sync_inf = _sync_inf + _self_mask * _reflec_inf

            # find successors
            _successors = self.successors(inputs_i)
            _successors_mask = torch.unsqueeze(
                torch.tensor(_successors, device=device), -1)
            _successors_mask = torch.broadcast_to(
                _successors_mask, (-1, -1, self.hidden_num))

            # propagation
            _prop_diff = torch.concat(
                (_next_self_state - _self_state, self.concept_embedding(inputs_i)), dim=-1)
            # _prop_diff = _next_self_state - _self_state

            # 1
            _prop_inf = self.f_prop(_prop_diff)
            _prop_inf = _successors_mask * \
                torch.broadcast_to(torch.unsqueeze(
                    _prop_inf, axis=1), (-1, self.ku_num, -1))
            # 2
            # _broadcast_diff = mx.nd.broadcast_to(mx.nd.expand_dims(_prop_diff, axis=1), (0, self.ku_num, 0))
            # _pro_inf = _successors_mask * self.f_prop(
            #     mx.nd.concat(_broadcast_diff, concept_embeddings, dim=-1)
            # )
            # _pro_inf = _successors_mask * self.f_prop(
            #     _broadcast_diff
            # )

            # aggregate
            _inf = self.f_agg(self.alpha * _sync_inf +
                              (1 - self.alpha) * _prop_inf)
            next_states, _ = self.rnn(_inf, [states])
            # next_states, _ = self.rnn(torch.concat((_inf, concept_embeddings), dim=-1), [states])
            # states = (1 - _self_mask) * next_states + _self_mask * _broadcast_next_self_states
            states = next_states
            output = self.sigmoid(torch.squeeze(
                self.out(self.dropout(states)), axis=-1))
            outputs.append(output)
            if valid_length is not None and not compressed_out:
                all_states.append([states])

        if valid_length is not None:
            if compressed_out:
                states = None
            outputs = mask_sequence_variable_length(
                torch, outputs, length, valid_length, axis, merge=True)

        return outputs, states
