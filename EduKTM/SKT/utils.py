# coding: utf-8
# 2021/8/16 @ tongshiwei
import torch
from torch import nn

__all__ = ["EAG"]


def indices2mask(indices, shape):
    """

    Parameters
    ----------
    indices: list of int
        M
    shape:
        (N, C, ...)

    Returns
    -------
    mask:
        (N, C, ...)

    Examples
    ---------
    >>> indices = [0, 2]
    >>> indices2mask(indices, (3, 2))
    tensor([[1., 1.],
            [0., 0.],
            [1., 1.]])
    """
    mask = torch.zeros(*shape)
    mask[indices] = torch.ones(*shape[1:])
    return mask


class EAG(nn.Module):
    """
    Erase-add gate

    Examples
    --------
    >>> import torch
    >>> eag = EAG()
    >>> src = torch.ones((2, 5, 3))
    >>> src
    tensor([[[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]]])
    >>> indices = [torch.tensor([0]), torch.tensor([1, 2])]
    >>> fea = [torch.tensor([[0., 1., 2.]]), torch.tensor([[4., 0., 1.], [5., 1., 2.]])]
    >>> eag(src, fea, indices)
    tensor([[[0., 1., 2.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [4., 0., 1.],
             [5., 1., 2.],
             [1., 1., 1.],
             [1., 1., 1.]]])

    >>> indices = torch.tensor([0, 2])
    >>> fea = torch.tensor([[0., 1., 2.], [5., 1., 2.]])
    >>> eag(src, fea, indices, single=True)
    tensor([[[0., 1., 2.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.],
             [5., 1., 2.],
             [1., 1., 1.],
             [1., 1., 1.]]])

    References
    ----------
    [1] Zhang J, Shi X, King I, et al.
    Dynamic key-value memory networks for knowledge tracing[C]//Proceedings of the 26th international conference
    on World Wide Web. 2017: 765-774.
    """

    def __init__(self):
        super(EAG, self).__init__()

    def forward(self, src, tar, indices, single=False):
        """

        Parameters
        ----------
        src: tensor
            (B, N, C, ...)
        tar: list of tensor or tensor
            (B, M, C, ...)
        indices: list of tensor or tensor
            (B, M, I)
        single

        Returns
        -------

        """
        tensor = []
        indices = torch.unsqueeze(indices, 1) if single else indices
        tar = torch.unsqueeze(tar, 1) if single else tar
        for _src, _tar, index in zip(src, tar, indices):
            erase = (1 - indices2mask(index, _src.shape)) * _src
            tensor.append(torch.index_add(erase, 0, index, _tar))
        return torch.stack(tensor)
