# coding: utf-8
# 2023/3/17 @ weizhehuang0827
from EduKTM import SKT
from EduKTM.SKT.utils import Graph


def test_train(data, conf, graphs, tmp_path):
    ku_num, hidden_num = conf
    mgkt = SKT(ku_num, graphs, hidden_num)
    mgkt.train(data, test_data=data, epoch=1)
    filepath = tmp_path / "skt.params"
    mgkt.save(filepath)
    mgkt.load(filepath)


def test_graph(conf, graphs_2):
    ku_num, _ = conf
    graph = Graph.from_file(ku_num, graphs_2)
    graph.info
    graph.neighbors(0, excluded=[1])
    none_graph = Graph(ku_num, [], [])
    none_graph.neighbors(0)
    none_graph.successors(0)
