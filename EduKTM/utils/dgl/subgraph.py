# coding: utf-8
# 2021/6/3 @ tongshiwei
import dgl
from dgl import NID, EID
import torch
from dgl.utils import extract_node_subframes_for_block, extract_edge_subframes, set_new_frames
from .const import FID
from queue import Queue
import threading as mp
import itertools as it
from torch.multiprocessing import Queue as TQueue, Process


def repr_block(block):
    return block.srcdata[dgl.NID], block.dstdata[dgl.NID], block.edges()


class OutBlockSampler(object):
    """
    Message Flow Graph (MFG)

    Notes
    -----
    src nodes will be included in dst nodes

    Examples
    --------
    >>> sampler = OutBlockSampler(2)
    >>> g = dgl.graph([])
    >>> g.add_nodes(10)
    >>> g.add_edges([1, 1, 1, 2, 3], [2, 4, 3, 4, 5])
    >>> blocks = sampler.sample_blocks(g, [1])
    >>> blocks
    [Block(num_src_nodes=1, num_dst_nodes=4, num_edges=3), Block(num_src_nodes=2, num_dst_nodes=4, num_edges=2)]
    >>> repr_block(blocks[0])
    (tensor([1]), tensor([1, 2, 3, 4]), (tensor([0, 0, 0]), tensor([1, 2, 3])))
    >>> repr_block(blocks[1])
    (tensor([2, 3]), tensor([2, 3, 4, 5]), (tensor([0, 1]), tensor([2, 3])))
    >>> blocks = sampler.sample_blocks(g, [1], src_in_seeds=True)
    >>> blocks
    [Block(num_src_nodes=1, num_dst_nodes=4, num_edges=3), Block(num_src_nodes=3, num_dst_nodes=5, num_edges=5)]
    >>> repr_block(blocks[0])
    (tensor([1]), tensor([1, 2, 3, 4]), (tensor([0, 0, 0]), tensor([1, 2, 3])))
    >>> repr_block(blocks[1])
    (tensor([1, 2, 3]), tensor([1, 2, 3, 4, 5]), (tensor([0, 0, 0, 1, 2]), tensor([1, 3, 2, 3, 4])))
    """

    def __init__(self, num_layers):
        self.num_layers = num_layers

    def sample_blocks(self, g, seed_nodes: list, src_in_seeds=False, *args) -> list:
        """

        Parameters
        ----------
        g
        seed_nodes: list
        src_in_seeds
        args

        Returns
        -------
        blocks: list
        """
        blocks = []
        for _ in range(self.num_layers):
            osg = dgl.subgraph.out_subgraph(g, seed_nodes)
            new_edges = osg.edata[EID]
            esg = dgl.edge_subgraph(g, new_edges)
            src_node_ids, dst_node_ids = osg.edges()
            edge_ids = [esg.edata[EID]]
            edge_frames = extract_edge_subframes(g, edge_ids)
            block = dgl.create_block(esg.edges())

            if src_in_seeds is False:
                node_frames = extract_node_subframes_for_block(
                    g, [torch.unique(src_node_ids)], [torch.unique(dst_node_ids)]
                )
                set_new_frames(block, node_frames=node_frames, edge_frames=edge_frames)
                seed_nodes = {ntype: block.dstnodes[ntype].data[NID] for ntype in block.dsttypes}

            node_frames = extract_node_subframes_for_block(
                g, [torch.unique(src_node_ids)],
                [torch.unique(torch.cat([dst_node_ids, src_node_ids], dim=-1))]
            )
            set_new_frames(block, node_frames=node_frames, edge_frames=edge_frames)
            blocks.append(block)
            if src_in_seeds is True:
                seed_nodes = {ntype: block.dstnodes[ntype].data[NID] for ntype in block.dsttypes}

        return blocks


class MFGLoader(object):
    def __init__(self, graph: dgl.DGLGraph, hop: int):
        self.graph = graph
        self.hop = hop
        self.sampler = OutBlockSampler(self.hop)

    def __call__(self, ids, features, mask=None):
        mask = it.cycle([None]) if mask is None else mask
        for nid_list, feature, valid in zip(ids, features, mask):
            if valid is False:
                yield []
            else:
                self.graph.ndata[FID] = feature
                yield self.sampler.sample_blocks(self.graph, nid_list)


def sample_blocks(sampler, graph, ids, features, mask, queue: Queue):
    for nid_list, feature, valid in zip(ids, features, mask):
        if valid is False:
            queue.put([], block=True)
        else:
            graph.ndata[FID] = feature
            queue.put(sampler.sample_blocks(graph, nid_list), block=True)
    # queue.put(StopIteration(), block=True)


class ParallelMFGLoader(object):
    def __init__(self, graph: dgl.DGLGraph, hop: int):
        self.graph = graph
        self.hop = hop
        self.sampler = OutBlockSampler(self.hop)
        self.process = None
        self.out_queue = None

    def __call__(self, ids, features, mask=None):
        mask = it.cycle([None]) if mask is None else mask
        if self.process is not None:
            self.process.join()
            self.process = None

        self.out_queue = Queue(4)
        self.process = mp.Thread(
            target=sample_blocks,
            args=(self.sampler, self.graph, ids, features, mask, self.out_queue),
            daemon=True
        )
        self.process.start()
        e = self.out_queue.get(block=True)
        while not isinstance(e, StopIteration):
            yield e
            e = self.out_queue.get(block=True)
        self.process.join()
        self.process = None


class TorchParallelMFGLoader(object):
    def __init__(self, graph: dgl.DGLGraph, hop: int):
        self.graph = graph
        self.hop = hop
        self.sampler = OutBlockSampler(self.hop)
        self.process = None
        self.out_queue = None

    def __call__(self, ids, features, mask=None):
        mask = it.cycle([None]) if mask is None else mask
        if self.process is not None:
            self.process.join()
            self.process = None

        self.out_queue = TQueue(4)
        self.process = Process(
            target=sample_blocks,
            args=(self.sampler, self.graph, ids, features, mask, self.out_queue),
            daemon=True
        )
        self.process.start()
        e = self.out_queue.get(block=True)
        while not isinstance(e, StopIteration):
            yield e
            e = self.out_queue.get(block=True)
        self.process.join()
        self.process = None


if __name__ == '__main__':

    # from torch import nn
    # import torch.nn.functional as F
    #
    # g = dgl.graph([])
    # node_num = 100
    # g.add_nodes(node_num)
    # g.add_edges([1, 1, 1, 2, 3, 5], [2, 4, 3, 4, 5, 9])
    #
    #
    # class StochasticTwoLayerGCN(nn.Module):
    #     def __init__(self, in_features, hidden_features, out_features):
    #         super().__init__()
    #         self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
    #         self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
    #
    #     def forward(self, blocks, x):
    #         x = F.relu(self.conv1(blocks[0], x))
    #         x = F.relu(self.conv2(blocks[1], x))
    #         return x

    #
    from dgl.dataloading import MultiLayerFullNeighborSampler

    #
    # sampler = OutBlockSampler(2)
    sampler = MultiLayerFullNeighborSampler(2)
    # # #
    # # # model = StochasticTwoLayerGCN(5, 20, 5)
    # # # opt = torch.optim.Adam(model.parameters())
    # # # blocks = sampler.sample_blocks(g, [1])
    # # # print([repr_block(block)[1] for block in blocks])
    # # # input_features = blocks[0].srcdata['features']
    # # # output_labels = blocks[-1].dstdata['label']
    # # # output_predictions = model(blocks, input_features)
    # # # print(output_predictions)
    # # #
    # # # g.ndata["features"] = torch.ones([10, 5])
    # # # g.ndata["label"] = torch.randint(1, [10])
    # # # blocks = sampler.sample_blocks(g, [1])
    # # # input_features = blocks[0].srcdata['features']
    # # # output_predictions = model(blocks, input_features)
    # # # g.ndata["features"][blocks[1].dstdata[dgl.NID]] = output_predictions
    # # # print(g.ndata["features"])
    # # # print(output_predictions)
    # #
    import dgl.function as fn

    #
    # #
    g = dgl.graph([])
    node_num = 900
    g.add_nodes(node_num)
    g.add_edges([1, 1, 1, 2, 3], [2, 4, 3, 4, 5])
    g = g.add_self_loop()
    g.ndata["features"] = torch.ones([node_num, 5])
    g.ndata["label"] = torch.randint(1, [node_num])
    g.edata["a"] = torch.ones(g.num_edges())
    # print(g.edata["a"])

    from longling import Clock

    from tqdm import tqdm

    print("testing")
    with Clock():
        for _ in tqdm(range(1000)):
            blocks = sampler.sample_blocks(g, [1])

# input_features = blocks[0].srcdata['features']
# print(input_features)
# print(blocks[0].edata["a"])
# print(blocks[0])
# print(repr_block(blocks[0]))
# blocks[0].update_all(fn.copy_src('features', 'm'), fn.sum('m', 'src_fea'))
# print(nn.Linear(10, 3)(torch.cat([blocks[0].dstdata["features"], blocks[0].dstdata["src_fea"]], dim=-1)))
