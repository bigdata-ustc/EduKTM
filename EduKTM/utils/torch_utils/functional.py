# coding: utf-8
# 2021/5/24 @ tongshiwei
__all__ = ["pick", "tensor2list", "length2mask", "get_sequence_mask", "sequence_mask"]

import torch
from torch import Tensor


def pick(tensor, index, axis=-1):
    return torch.gather(tensor, axis, index.unsqueeze(axis)).squeeze(axis)


def tensor2list(tensor: Tensor):
    return tensor.cpu().tolist()


def length2mask(length, max_len, valid_mask_val, invalid_mask_val):
    mask = []

    if isinstance(valid_mask_val, Tensor):
        valid_mask_val = tensor2list(valid_mask_val)
    if isinstance(invalid_mask_val, Tensor):
        invalid_mask_val = tensor2list(invalid_mask_val)
    if isinstance(length, Tensor):
        length = tensor2list(length)

    for _len in length:
        mask.append([valid_mask_val] * _len + [invalid_mask_val] * (max_len - _len))

    return torch.tensor(mask)


def get_sequence_mask(shape, sequence_length, axis=1):
    assert axis <= len(shape)
    mask_shape = shape[axis + 1:]

    valid_mask_val = torch.ones(mask_shape)
    invalid_mask_val = torch.zeros(mask_shape)

    max_len = shape[axis]

    return length2mask(sequence_length, max_len, valid_mask_val, invalid_mask_val)


def sequence_mask(tensor: Tensor, sequence_length, axis=1):
    mask = get_sequence_mask(tensor.shape, sequence_length, axis).to(tensor.device)
    return tensor * mask
