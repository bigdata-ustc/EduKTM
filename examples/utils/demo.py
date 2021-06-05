# coding: utf-8
# 2021/6/5 @ tongshiwei

import dgl
import torch
from dgl import NID, EID
from EduKTM.utils import OutBlockSampler

g = dgl.graph([])

g.add_nodes(10)
g.add_edges(
    [1, 4, 3], [3, 2, 5]
)
g.add_edges(4, [7, 8, 9])
g.add_edges([2, 6, 8], 9)
g.ndata["features"] = torch.randn([10, 2])
g.ndata["label"] = torch.randint(1, [10])
print(g.ndata["features"])
print(g)
print(g.edges())
og = dgl.subgraph.out_subgraph(g, [4])
print(og.edges())
new_edges = og.edata[EID]
print(new_edges)
es = dgl.edge_subgraph(g, new_edges)
print(es.edges())
print(es.nodes())
print(es.srcnodes())
print(es.srcdata[NID])
print(es.dstnodes())
print(es.dstdata[NID])
print("----------------")
obs = OutBlockSampler(3)
print(obs.sample_blocks(g, [4]))
