# coding: utf-8
# 2021/6/3 @ tongshiwei
import dgl
from dgl import NID, EID
import torch
from dgl.utils import extract_node_subframes_for_block, extract_edge_subframes, set_new_frames


def repr_block(block):
    return block.srcdata[dgl.NID], block.dstdata[dgl.NID], block.edges()


class OutBlockSampler(object):
    """
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
    (tensor([1]), tensor([2, 3, 4]), (tensor([0, 0, 0]), tensor([1, 2, 3])))
    >>> repr_block(blocks[1])
    (tensor([2, 3]), tensor([4, 5]), (tensor([0, 1]), tensor([2, 3])))
    """

    def __init__(self, num_layers):
        self.num_layers = num_layers

    def sample_blocks(self, g, seed_nodes, *args):
        """

        Parameters
        ----------
        g
        seed_nodes: list
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
            block = dgl.create_block(esg.edges())
            src_node_ids, dst_node_ids = osg.edges()
            edge_ids = [esg.edata[EID]]
            node_frames = extract_node_subframes_for_block(
                g, [torch.unique(src_node_ids)], [torch.unique(dst_node_ids)]
            )
            edge_frames = extract_edge_subframes(g, edge_ids)
            set_new_frames(block, node_frames=node_frames, edge_frames=edge_frames)
            seed_nodes = {ntype: block.dstnodes[ntype].data[NID] for ntype in block.dsttypes}
            blocks.append(block)
        return blocks
