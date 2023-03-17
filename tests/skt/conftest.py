# coding: utf-8
# 2023/3/17 @ weizhehuang0827
import pytest
import json
from EduKTM.utils.tests import pseudo_data_generation
from EduKTM.SKT.etl import transform


@pytest.fixture(scope="package")
def conf():
    ques_num = 10
    hidden_num = 5
    return ques_num, hidden_num


@pytest.fixture(scope="package")
def data(conf):
    ques_num, _ = conf
    return transform(pseudo_data_generation(ques_num), 16)


@pytest.fixture(scope="session")
def graphs(tmpdir_factory):
    graph_dir = tmpdir_factory.mktemp("data").join("graph_1.json")
    _graphs = []
    with open(graph_dir, 'w') as wf:
        json.dump([[0, 1], [0, 2]], wf)
    _graphs.append([graph_dir, False])
    _graphs.append([graph_dir, True])
    graph_dir = tmpdir_factory.mktemp("data").join("graph_2.json")
    with open(graph_dir, 'w') as wf:
        json.dump([[1, 2, 0.9], [1, 5, 0.1]], wf)
    _graphs.append([graph_dir, True, 0.5])
    return _graphs
