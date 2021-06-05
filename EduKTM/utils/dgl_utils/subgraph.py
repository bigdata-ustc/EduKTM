# coding: utf-8
# 2021/6/3 @ tongshiwei
import dgl
from dgl import NID, EID
import torch
from dgl.utils import extract_node_subframes_for_block, extract_edge_subframes, set_new_frames


class OutBlockSampler(object):
    def __init__(self, num_layers):
        self.num_layers = num_layers

    def sample_blocks(self, g, seed_nodes, *args):
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


# print(es)
# b = dgl.create_block(es.edges())
# print(b)
# nsf = extract_node_subframes_for_block(g, [torch.unique(og.edges()[0])], [torch.unique(og.edges()[1])])
# print(nsf)
# esf = extract_edge_subframes(g, [es.edata[EID]])
# print(esf)
# set_new_frames(b, node_frames=nsf, edge_frames=esf)
# print(b)
# print(b.srcdata)
# print(b.dstdata)
# sg = g.out_subgraph([4])
# print(sg)
#
# print_block(dgl.create_block(sg.edges()))


