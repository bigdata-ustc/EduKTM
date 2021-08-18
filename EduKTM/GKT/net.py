# coding: utf-8
# 2021/5/24 @ tongshiwei

import torch
from torch import nn
import dgl
from EduKTM.utils.dgl import ParallelMFGLoader as MFGLoader, FID
from EduKTM.SKT.utils import EAG
import dgl.function as fn
import itertools as it
from EduKTM.utils import batch_pick
from longling import Clock


class StatNeighbor(nn.Module):
    """
    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(0)
    >>> from EduKTM.utils.dgl import FID
    >>> g = dgl.graph([])
    >>> g.add_nodes(10)
    >>> g.add_edges([1, 1, 1, 2, 3], [2, 4, 3, 4, 5])
    >>> g.ndata[FID] = torch.ones([10, 5])
    >>> g.edata["out_a"] = torch.ones(g.num_edges())
    >>> g.edata["in_a"] = torch.ones(g.num_edges())
    >>> neighbor_fn = StatNeighbor(5)
    >>> neighbor_fn(g)
    tensor([[-1.0059, -0.1781,  0.2778, -0.7120, -0.7661],
            [-1.0059, -0.1781,  0.2778, -0.7120, -0.7661],
            [-0.2690,  0.0981, -0.0044, -0.3191, -0.7040],
            [-0.2690,  0.0981, -0.0044, -0.3191, -0.7040],
            [ 0.4680,  0.3744, -0.2866,  0.0739, -0.6418],
            [-0.2690,  0.0981, -0.0044, -0.3191, -0.7040],
            [-1.0059, -0.1781,  0.2778, -0.7120, -0.7661],
            [-1.0059, -0.1781,  0.2778, -0.7120, -0.7661],
            [-1.0059, -0.1781,  0.2778, -0.7120, -0.7661],
            [-1.0059, -0.1781,  0.2778, -0.7120, -0.7661]], grad_fn=<AddBackward0>)
    """

    def __init__(self, features_num):
        super().__init__()
        self.features_num = features_num
        self.fc_in = nn.Linear(features_num * 2, features_num)
        self.fc_out = nn.Linear(features_num * 2, features_num)

    def forward(self, blocks: (dgl.DGLGraph, list)):
        block = blocks[0] if isinstance(blocks, list) else blocks
        block.update_all(fn.copy_src(FID, 'm'), fn.sum('m', 'src_fea'))
        fea = torch.cat([block.dstdata[FID], block.dstdata["src_fea"]], dim=-1)
        in_fea = self.fc_in(fea)
        in_weight = block.edata["in_a"]
        out_fea = self.fc_out(fea)
        out_weight = block.edata["out_a"]
        return out_weight * out_fea + in_weight * in_fea


class GKTNet(nn.Module):
    """
    Examples
    ---------
    >>> import torch
    >>> g = dgl.graph([])
    >>> g.add_nodes(10)
    >>> g.add_edges([1, 1, 1, 2, 3], [2, 4, 3, 4, 5])
    >>> g = g.add_self_loop()
    >>> g.ndata[FID] = torch.ones([10, 5])
    >>> g.edata["out_a"] = torch.ones(g.num_edges(), 1)
    >>> g.edata["in_a"] = torch.ones(g.num_edges(), 1)
    >>> gkt = GKTNet(g, 10, 5, 3)
    >>> item_ids = torch.tensor([[1, 3, 5], [0, 2, 4]])
    >>> responses = torch.tensor([[2, 7, 11], [1, 4, 9]])
    >>> outputs, hn = gkt(item_ids, responses)
    >>> outputs.shape
    torch.Size([2, 3, 10])
    >>> hn.shape
    torch.Size([2, 10, 3])
    >>> next_item_ids = torch.tensor([[0, 2, 4], [1, 3, 5]])
    >>> outputs, _ = gkt(item_ids, responses, next_item_ids=next_item_ids)
    >>> outputs.shape
    torch.Size([2, 3])
    """

    def __init__(self, graph: dgl.DGLGraph, item_size, item_dim, state_dim):
        """

        Parameters
        ----------
        graph: dgl.DGLGraph
            must have self loop
        item_size
        item_dim
        state_dim

        """
        super(GKTNet, self).__init__()
        self.graph = graph
        self.item_size = item_size
        self.state_dim = state_dim
        self.item_embedding = torch.nn.Embedding(item_size, item_dim)
        self.response_embedding = torch.nn.Embedding(item_size * 2, item_dim)
        self.sampler = MFGLoader(graph, 1)
        self.eag = EAG()
        self.feature_num = item_dim + state_dim
        self.neighbor = StatNeighbor(self.feature_num)
        self.rnn = torch.nn.GRUCell(item_dim + state_dim, state_dim)
        self.fc = nn.Linear(state_dim, 1)
        self.sig = nn.Sigmoid()

    def get_states(self, batch_size):
        h0 = torch.zeros(batch_size, self.item_size, self.state_dim)
        return h0

    def get_memory(self, batch_size):
        m0 = torch.zeros(batch_size, self.item_size, self.feature_num)
        return m0

    def get_item_embedding_weight(self, batch_size):
        return torch.unsqueeze(self.item_embedding(torch.arange(self.item_size)), 0).repeat(batch_size, 1, 1)

    def forward(self, item_ids, responses, mask=None, next_item_ids=None):
        """

        Parameters
        ----------
        item_ids: tensor
            (batch, seq, item_id)

        responses: tensor
            (batch, seq, response_id)

        mask: tensor
            (batch, length)

        next_item_ids: tensor
            (batch, seq, item_id)

        Returns
        -------

        """
        batch_size = item_ids.shape[0]
        h = self.get_states(batch_size)  # (batch, N, state_dim)
        m = self.get_memory(batch_size)  # (batch, N, state_dim)
        item_ids = torch.transpose(item_ids, 0, 1)  # (seq, batch, item_id)
        items = self.item_embedding(item_ids)  # (seq, batch, item_dim)
        responses = self.response_embedding(torch.transpose(responses, 0, 1))  # (seq, batch, item_dim)
        outputs = []
        clock = Clock()
        clock1 = Clock()
        clock2 = Clock()
        gt = 0
        gt2 = 0

        next_item_ids = it.cycle([None]) if next_item_ids is None else torch.transpose(next_item_ids, 0, 1)

        clock.start()
        for t, (item_id, item, response, next_item_id) in enumerate(
                zip(item_ids, items, responses, next_item_ids)
        ):  # iterate on time
            # (batch, id) for item_id, next_item_id, (batch, item_dim) for item and response
            ie = self.get_item_embedding_weight(batch_size)  # (batch, N, item_dim)
            h_ = torch.cat([h, self.eag(ie, response, item_id, single=True)], dim=-1)
            m_tilde = []
            _mask = None if mask is None else t <= mask
            clock1.start()
            for b, (m_, blocks) in enumerate(
                    zip(m, self.sampler(torch.unsqueeze(item_id, -1), h_, _mask))
            ):  # iterate on batch
                if not blocks:
                    # masked, no need to operate on the graph
                    m_tilde.append(m_)
                else:
                    dst_nodes = blocks[-1].dstdata[dgl.NID]
                    clock2.start()
                    dst_fea = self.neighbor(blocks)
                    gt2 += clock2.end()
                    m_tilde.append(self.eag([m_], [dst_fea], [dst_nodes]))
            gt += clock1.end()
            m = torch.cat(m_tilde)
            next_h = self.rnn(
                torch.reshape(m, (-1, m.shape[-1])),
                torch.reshape(h, (-1, h.shape[-1])),
            )

            next_h = torch.reshape(next_h, (batch_size, self.item_size, next_h.shape[-1]))
            if mask is not None:
                _mask = _mask.reshape(_mask.shape[0], 1, 1)
                h = next_h * _mask + h * ~_mask
            else:
                h = next_h

            if next_item_id is not None:
                state = batch_pick(h, next_item_id)
            else:
                state = h
            outputs.append(self.sig(self.fc(state)).squeeze(-1))

        return torch.transpose(torch.stack(outputs), 0, 1), h
