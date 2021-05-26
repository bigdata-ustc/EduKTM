# coding: utf-8
# 2021/5/26 @ tongshiwei
import pytest

from EduKTM.utils.tests import pseudo_data_generation
from EduKTM.DKTPlus.etl import transform


@pytest.fixture(scope="package")
def conf():
    ques_num = 10
    hidden_num = 10
    return ques_num, hidden_num


@pytest.fixture(scope="package")
def data(conf):
    ques_num, _ = conf
    return transform(pseudo_data_generation(ques_num), 32)
