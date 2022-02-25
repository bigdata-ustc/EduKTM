import pytest
from EduKTM.utils.tests import pseudo_data_generation
from EduKTM.GKT.etl import transform

@pytest.fixture(scope="package")
def conf():
    ques_num = 124
    hidden_num = 5
    graph = "../../data/assistment_2009_2010/transition_graph.json"
    return ques_num, graph, hidden_num


@pytest.fixture(scope="package")
def data(conf):
    ques_num, _, _ = conf
    return transform(pseudo_data_generation(ques_num), 16)
