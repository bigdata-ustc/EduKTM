# coding: utf-8
# 2022/3/18 @ ouyangjie


import pytest
from EduKTM import DKVMN


@pytest.mark.parametrize("lr", [0, 0.1])
@pytest.mark.parametrize("maxgradnorm", [-5, 5])
def test_train(data, conf, tmp_path, lr, maxgradnorm):
    n_question, batch_size = conf

    params = {
        'max_iter': 2,
        'lr': lr,
        'final_fc_dim': 5,
        'key_embedding_dim': 5,
        'batch_size': batch_size,
        'value_embedding_dim': 10,
        'memory_size': 5,
        'n_question': n_question,
        'seqlen': 10,
        'maxgradnorm': maxgradnorm
    }

    dkvmn = DKVMN(n_question=params['n_question'],
                  batch_size=params['batch_size'],
                  key_embedding_dim=params['key_embedding_dim'],
                  value_embedding_dim=params['value_embedding_dim'],
                  memory_size=params['memory_size'],
                  key_memory_state_dim=params['key_embedding_dim'],
                  value_memory_state_dim=params['value_embedding_dim'],
                  final_fc_dim=params['final_fc_dim'])

    dkvmn.train(params, data, test_data=data)
    filepath = tmp_path / "dkvmn.params"
    dkvmn.save(filepath)
    dkvmn.load(filepath)
    dkvmn.eval(params, data)
