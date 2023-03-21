# coding: utf-8
# 2022/2/25 @ fannazya
# Code reused from https://github.com/tswsxk/Baize
__all__ = ["GRUCell", "begin_states", "get_states", "expand_tensor",
           "mask_sequence_variable_length", "format_sequence"]

import torch
from torch import nn


def as_list(obj) -> list:
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)


def begin_states(shapes, func=torch.zeros):
    states = []
    for i, shape in enumerate(as_list(shapes)):
        state = func(shape)
        states.append(state)
    return states


def get_states(indexes, states):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.numpy().tolist()
    if isinstance(indexes, list):
        return torch.stack([*[get_states(index, state) for (index, state) in zip(indexes, states)]])
    elif isinstance(indexes, (int, float)):
        return states[int(indexes)]


def sequence_mask(X, valid_len, value):
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value

    return X


def mask_sequence_variable_length(F, data, valid_length):
    assert valid_length is not None
    if not isinstance(data, torch.Tensor):
        data = F.stack([*data], dim=1)
    outputs = sequence_mask(
        data, valid_len=valid_length,
        value=0
    )
    return outputs


def format_sequence(length, inputs, layout, merge, in_layout=None):

    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    batch_axis = layout.find('N')
    batch_size = 0
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, torch.Tensor):
        batch_size = inputs.shape[batch_axis]
        if merge is False:
            assert length is None or length == inputs.shape[in_axis]
            split_size = [1] * inputs.shape[in_axis]
            inputs = as_list(torch.split(inputs, split_size_or_sections=split_size, dim=in_axis))

    return inputs, axis, batch_size


def expand_tensor(tensor, expand_axis, expand_num):
    assert len(tensor.shape) == 2

    _tensor = torch.unsqueeze(tensor, expand_axis)

    shape = list(_tensor.shape)
    shape[expand_axis] = expand_num
    _tensor = torch.broadcast_to(_tensor, tuple(shape))

    return _tensor


class GRUCell(nn.Module):
    def __init__(self, hidden_num):
        super(GRUCell, self).__init__()
        self.i2h = nn.Linear(hidden_num, 3 * hidden_num)
        self.h2h = nn.Linear(hidden_num, 3 * hidden_num)
        self.reset_act = nn.Sigmoid()
        self.update_act = nn.Sigmoid()
        self.act = nn.Tanh()

    def forward(self, inputs, states):
        prev_state_h = states[0]

        i2h = self.i2h(inputs)
        h2h = self.h2h(prev_state_h)
        i2h_r, i2h_z, i2h = torch.chunk(i2h, 3, dim=-1)
        h2h_r, h2h_z, h2h = torch.chunk(h2h, 3, dim=-1)

        reset_gate = self.reset_act(i2h_r + h2h_r)
        update_gate = self.update_act(i2h_z + h2h_z)
        next_h_tmp = self.act(i2h + reset_gate * h2h)
        ones = torch.ones_like(update_gate)
        next_h = (ones - update_gate) * next_h_tmp + update_gate * prev_state_h

        return next_h, [next_h]
