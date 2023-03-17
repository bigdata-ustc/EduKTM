# coding: utf-8
# 2023/3/17 @ weizhehuang0827


import json
import torch
import networkx as nx
__all__ = ["Graph", "load_graph"]


def as_list(obj) -> list:
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


class Graph(object):
    def __init__(self, ku_num, directed_graphs, undirected_graphs):
        self.ku_num = ku_num
        self.directed_graphs = as_list(directed_graphs)
        self.undirected_graphs = as_list(undirected_graphs)

    @staticmethod
    def _info(graph: nx.Graph):
        return {"edges": len(graph.edges)}

    @property
    def info(self):
        return {
            "directed": [self._info(graph) for graph in self.directed_graphs],
            "undirected": [self._info(graph) for graph in self.undirected_graphs]
        }

    def neighbors(self, x, ordinal=True, merge_to_one=True, with_weight=False, excluded=None):
        excluded = set() if excluded is None else excluded

        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, list):
            return [self.neighbors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                if len(self.undirected_graphs) == 0:
                    return None if not merge_to_one else []
                elif len(self.undirected_graphs) == 1:
                    return [v for v in self.undirected_graphs[0].neighbors(int(x)) if v not in excluded]
                else:
                    if not merge_to_one:
                        return [[v for v in graph.neighbors(int(x)) if v not in excluded] for graph in
                                self.undirected_graphs]
                    else:
                        _ret = []
                        for graph in self.undirected_graphs:
                            _ret.extend([v for v in graph.neighbors(
                                int(x)) if v not in excluded])
                        return _ret
            else:  # ordinal
                if not merge_to_one:
                    if len(self.undirected_graphs) == 0:
                        return None
                    elif len(self.undirected_graphs) == 1:
                        graph = self.undirected_graphs[0]
                        _ret = [0] * self.ku_num
                        for i in graph.neighbors(int(x)):
                            if i in excluded:
                                continue
                            if with_weight:
                                _ret[i] = graph[x][i].get('weight', 1)
                            else:
                                _ret[i] = 1
                        return _ret
                    else:
                        _ret = []
                        for graph in self.undirected_graphs:
                            __ret = [0] * self.ku_num
                            for i in graph.neighbors(int(x)):
                                if i in excluded:
                                    continue
                                if with_weight:
                                    __ret[i] = graph[x][i].get('weight', 1)
                                else:
                                    __ret[i] = 1
                            _ret.append(__ret)
                else:
                    if len(self.undirected_graphs) == 0:
                        return [0] * self.ku_num
                    else:
                        _ret = [0] * self.ku_num
                        for graph in self.undirected_graphs:
                            for i in graph.neighbors(int(x)):
                                if i in excluded:
                                    continue
                                if with_weight:
                                    _ret[i] += graph[x][i].get('weight', 1)
                                else:
                                    _ret[i] = 1
                        return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    def successors(self, x, ordinal=True, merge_to_one=True, excluded=None):
        excluded = set() if excluded is None else excluded

        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, list):
            return [self.neighbors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                if len(self.directed_graphs) == 0:
                    return None if not merge_to_one else []
                elif len(self.directed_graphs) == 1:
                    return [v for v in self.directed_graphs[0].successors(int(x)) if v not in excluded]
                else:
                    if not merge_to_one:
                        return [[v for v in graph.successors(int(x)) if v not in excluded] for graph in
                                self.directed_graphs]
                    else:
                        _ret = []
                        for graph in self.directed_graphs:
                            _ret.extend([v for v in graph.successors(
                                int(x)) if v not in excluded])
                        return _ret
            else:
                if not merge_to_one:
                    if len(self.directed_graphs) == 0:
                        return None
                    elif len(self.directed_graphs) == 1:
                        _ret = [0] * self.ku_num
                        for i in self.directed_graphs[0].successors(int(x)):
                            if i in excluded:
                                continue
                            _ret[i] = 1
                        return _ret
                    else:
                        _ret = []
                        for graph in self.directed_graphs:
                            __ret = [0] * self.ku_num
                            for i in graph.successors(int(x)):
                                if i in excluded:
                                    continue
                                _ret[i] = 1
                            _ret.append(__ret)
                else:
                    if len(self.directed_graphs) == 0:
                        return [0] * self.ku_num
                    else:
                        _ret = [0] * self.ku_num
                        for graph in self.directed_graphs:
                            for i in graph.successors(int(x)):
                                if i in excluded:
                                    continue
                                _ret[i] = 1
                        return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    @classmethod
    def from_file(cls, graph_nodes_num, graph_params):
        directed_graphs = []
        undirected_graphs = []
        for graph_param in graph_params:
            graph, directed = load_graph(
                graph_nodes_num, *as_list(graph_param))
            if directed:
                directed_graphs.append(graph)
            else:
                undirected_graphs.append(graph)
        return cls(graph_nodes_num, directed_graphs, undirected_graphs)


def load_graph(graph_nodes_num, filename=None, directed: bool = True, threshold=0.0):
    directed = bool(directed)
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(range(graph_nodes_num))
    if threshold < 0.0:
        for i in range(graph_nodes_num):
            for j in range(graph_nodes_num):
                graph.add_edge(i, j)
    else:
        assert filename is not None
        with open(filename) as f:
            for data in json.load(f):
                pre, suc = data[0], data[1]
                if len(data) >= 3 and float(data[2]) < threshold:
                    continue
                elif len(data) >= 3:
                    weight = float(data[2])
                    graph.add_edge(pre, suc, weight=weight)
                    continue
                graph.add_edge(pre, suc)
    return graph, directed
