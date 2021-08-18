# coding: utf-8
# 2021/5/24 @ tongshiwei
__all__ = ["batch_pick", "pick", "tensor2list", "length2mask", "get_sequence_mask", "sequence_mask"]

import torch
from torch import Tensor


def batch_pick(tensor, batch_index, keep_dim=False):
    """

    Parameters
    ----------
    tensor:
        (B, N, C)
    batch_index
        (B, )
    keep_dim

    Returns
    -------


    Examples
    --------
    >>> tensor = torch.tensor([[[0, 1, 2], [1, 11, 12]], [[2, 21, 22], [3, 31, 32]]])
    >>> index = torch.tensor([0, 1])
    >>> batch_pick(tensor, index)
    tensor([[ 0,  1,  2],
            [ 3, 31, 32]])
    >>> batch_pick(tensor, index, keep_dim=True)
    tensor([[[ 0,  1,  2]],
    <BLANKLINE>
            [[ 3, 31, 32]]])
    """
    batch_index = batch_index.reshape(batch_index.shape[0], 1, 1)
    batch_index = batch_index.repeat(1, 1, tensor.shape[-1])
    tensor = torch.gather(tensor, 1, batch_index)
    if keep_dim is False:
        return tensor.squeeze(1)
    return tensor


def pick(tensor, index, axis=-1):
    """

    Parameters
    ----------
    tensor
    index
    axis

    Returns
    -------

    Examples
    --------
    >>> import torch
    >>> tensor = torch.tensor([[[0, 1], [10, 11], [20, 21]], [[30, 31], [40, 41], [50, 51]]])
    >>> tensor
    tensor([[[ 0,  1],
             [10, 11],
             [20, 21]],
    <BLANKLINE>
            [[30, 31],
             [40, 41],
             [50, 51]]])
    """
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
