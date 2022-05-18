# coding: utf-8
# 2022/2/26 @ fannazya
import pytest
import json
from EduKTM.utils.tests import pseudo_data_generation
from EduKTM.GKT.etl import transform


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
def graph_file(tmpdir_factory):
    graph_dir = tmpdir_factory.mktemp("data").join("graph.json")
    with open(graph_dir, 'w') as wf:
        json.dump([[0, 1], [0, 2]], wf)
    return graph_dir
