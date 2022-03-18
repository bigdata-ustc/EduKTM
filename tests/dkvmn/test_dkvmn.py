# coding: utf-8
# 2022/3/18 @ ouyangjie

import pytest
from EduKTM import DKVMN

@pytest.mark.parametrize("lr", [0, 0.1])
def test_train(data, conf, tmp_path, lr):
    n_question, batch_size, lr = conf

    params = {
    'gpu': 0, # 'the gpu will be used, e.g "0,1,2,3"'
    'max_iter': 30, # 'number of iterations'
    'lr': lr, # 'initial learning rate'
    'final_fc_dim': 5, # 'hidden state dim for final fc layer'
    'key_embedding_dim': 5, # 'question embedding dimensions'
    'batch_size': batch_size, # 'the batch size'
    'value_embedding_dim': 20, # 'answer and question embedding dimensions'
    'memory_size': 5, # 'memory size'
    'n_question': n_question, # 'the number of unique questions in the dataset'
    'seqlen': 10, # 'the allowed maximum length of a sequence'
    }

    dkvmn = DKVMN(n_question=params['n_question'],
                  batch_size=params['batch_size'],
                  key_embedding_dim=params['key_embedding_dim'],
                  value_embedding_dim=params['value_embedding_dim'],
                  memory_size=params['memory_size'],
                  key_memory_state_dim=params['key_memory_state_dim'],
                  value_memory_state_dim=params['value_memory_state_dim'],
                  final_fc_dim=params['final_fc_dim'])

    dkvmn.train(params, data)
    
    filepath = tmp_path / "dkvmn.params"
    dkvmn.save(filepath)
    dkvmn.load(filepath)