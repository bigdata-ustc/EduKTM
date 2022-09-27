import random
import sys
import argparse

import torch
import numpy as np

from EduKTM import OKT
from load_data import DATA

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test OKT')
    parser.add_argument('--model', type=str, default='okt', help='optinal values: okt, okt_not, okt_nu, okt_ne')
    parser.add_argument('--dataset', type=str, default="assist2017", help='optinal values: assist2017, assist2012, nips2020_1_2')

    params = parser.parse_args()
    dataset_name = params.dataset
    model_name = params.model

    n_at, n_it, n_exercise, n_question = 0, 0, 0, 0
    has_at = True

    if dataset_name == 'assist2017':
        n_at = 1326
        n_it = 2873
        n_exercise = 3162
        n_question = 102
        seq_len = 500
        dataset_path = 'anonymized_full_release_competition_dataset'
    elif dataset_name == 'assist2012':
        n_at = 26411
        n_it = 29694
        n_exercise = 53091
        n_question = 265
        seq_len = 100
        dataset_path = '2012-2013-data-with-predictions-4-final'
    elif dataset_name == 'nips2020_1_2':
        n_it = 42148
        n_exercise = 27613
        n_question = 1125
        seq_len = 100
        dataset_path = 'NIPS2020/task_1_2'
        has_at = False
    else:
        raise Exception('no dataset named %s' % dataset_name)

    d_q, d_e = 32, 128
    d_p, d_a = 128, 128
    d_at = 50
    d_h = 128

    dropout = 0.3

    if dataset_name == 'assist2012':
        batch_size = 128
        lr = 1e-3
        lr_decay_step = 10
        lr_decay_rate = 0.1
        epoch = 10
    elif dataset_name == 'assist2017':
        batch_size = 64
        lr = 3e-3
        lr_decay_step = 10
        lr_decay_rate = 0.5
        epoch = 30
    elif dataset_name == 'nips2020_1_2':
        batch_size = 512
        lr = 2e-3
        lr_decay_step = 5
        lr_decay_rate = 0.5
        epoch = 20

    data_path = './data/' + dataset_path
    dat = DATA(seqlen=seq_len, separate_char=',', has_at=has_at)
    test_data = dat.load_data(data_path + '/test.txt')
    model_file_path = model_name + '-' + dataset_name + '.params'

    print('model %s is at dataset %s, and the model will save to path %s' % (model_name, dataset_name, model_file_path))

    # k-fold cross validation
    k = 5
    test_rmse_sum, test_r2_sum, test_auc_sum, test_accuracy_sum = .0, .0, .0, .0
    for i in range(k):
        model = OKT(n_at, n_it, n_exercise, n_question, d_e, d_q, d_a, d_at, d_p, d_h,
                    batch_size=batch_size, dropout=dropout)
        train_data = dat.load_data(data_path + '/train' + str(i) + '.txt')
        valid_data = dat.load_data(data_path + '/valid' + str(i) + '.txt')
        best_train_auc, best_valid_auc = model.train(train_data, valid_data,
                                                     epoch=epoch, lr=lr, lr_decay_step=lr_decay_step,
                                                     lr_decay_rate=lr_decay_rate,
                                                     filepath=model_file_path)
        print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
        # test
        model.load(model_file_path)
        _, test_rmse, test_r2, test_auc, test_accuracy = model.eval(test_data)
        print("[fold %d] rmse: %.6f, r2: %.6f, auc: %.6f, accuracy: %.6f" % (i, test_rmse, test_r2, test_auc, test_accuracy))
        test_rmse_sum += test_rmse
        test_r2_sum += test_r2
        test_auc_sum += test_auc
        test_accuracy_sum += test_accuracy
    print('%d-fold validation:' % k)
    print('avg of test data (RMSE, r2, auc, accuracy): %f, %f, %f, %f' % (
        test_rmse_sum / k, test_r2_sum / k, test_auc_sum / k, test_accuracy_sum / k))
