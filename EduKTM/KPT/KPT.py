# coding: utf-8
# 2021/4/5 @ liujiayu

import logging
import numpy as np
import pickle
from tqdm import tqdm
from collections import namedtuple
from collections import defaultdict
from EduKTM import KTM

hyper_para = namedtuple("hyperparameters", ["r", "D", "deltaT", "S", "lambda_U_1", "lambda_U", "lambda_P", "lambda_V"])
default_hyper = hyper_para(6, 2, 1, 5, 0.01, 2, 2, 0.01)  # lambda_V works as lambda_S in EKPT


def init_parameters(stu_num, prob_num, know_num, time_window_num):
    u_latent = np.random.normal(0.5, 0.01, size=(time_window_num, stu_num, know_num))
    i_latent = 0.1 * np.random.uniform(0, 1, size=(prob_num, know_num))  # problems' latent vector(V)
    alpha = np.random.uniform(0, 1, size=stu_num)
    B = 0.01 * np.random.normal(0, 1, size=prob_num)
    return u_latent, i_latent, alpha, B


def stu_curve(u_latent, alpha, r, D, deltaT, S, time_freq):  # learning and forgetting curve
    freq_norm = D * time_freq / (time_freq + r)
    learn_factor = u_latent * freq_norm
    forget_factor = u_latent * np.exp(-deltaT / S)
    pred_u = learn_factor * np.expand_dims(alpha, axis=1) + forget_factor * np.expand_dims(1 - alpha, axis=1)
    return pred_u, freq_norm


class KPT(KTM):
    """
    (E)KPT model, training (MAP) and testing methods
    Parameters
    ----------
    mode: str
        mode = 'KPT' or 'EKPT'
    q_m: array
        Q matrix, shape = (prob_num, know_num)
    stu_num: int
        number of students
    prob_num: int
        number of problems
    know_num: int
        number of knowledge
    time_window_num: int
        number of time windows
    args: namedtuple
        all hyper-parameters
    ----------
    """

    def __init__(self, mode, q_m, stu_num, prob_num, know_num, time_window_num, args=default_hyper):
        super(KPT, self).__init__()
        self.mode = mode
        self.args = args
        self.q_m = q_m
        self.stu_num, self.prob_num, self.know_num = stu_num, prob_num, know_num
        self.time_window_num = time_window_num
        self.u_latent, self.i_latent, self.alpha, self.B = init_parameters(stu_num, prob_num, know_num, time_window_num)
        # partial order of knowledge in each problem
        self.par_mat = np.zeros(shape=(prob_num, know_num, know_num))
        for i in range(prob_num):
            for o1 in range(know_num):
                if self.q_m[i][o1] == 0:
                    continue
                for o2 in range(know_num):
                    if self.q_m[i][o2] == 0:
                        self.par_mat[i][o1][o2] = 1

        # exercise relation (only used in EKPT)
        self.exer_neigh = (np.dot(self.q_m, self.q_m.transpose()) > 0).astype(int)

    def train(self, train_data, epoch, lr=0.001, lr_b=0.0001, epsilon=1e-3, init_method='mean') -> ...:
        # train_data(list): response data, length = time_window_num, e.g.[[{'user_id':, 'item_id':, 'score':},...],...]
        assert self.time_window_num == len(train_data), 'number of time windows conflicts'
        u_latent, i_latent = np.copy(self.u_latent), np.copy(self.i_latent)
        alpha, B = np.copy(self.alpha), np.copy(self.B)
        mode_ind = int(self.mode == 'EKPT')
        # mean score of each student in train_data
        sum_score = np.zeros(shape=self.stu_num)
        sum_count = np.zeros(shape=self.stu_num)

        # knowledge frequency in each time window
        time_freq = np.zeros(shape=(self.time_window_num, self.stu_num, self.know_num))
        for t in range(self.time_window_num):
            for record in train_data[t]:
                user, item, rating = record['user_id'], record['item_id'], record['score']
                time_freq[t][user][np.where(self.q_m[item] == 1)[0]] += 1
                sum_score[user] += rating
                sum_count[user] += 1

        # initialize student latent with mean score
        if init_method == 'mean':
            u_latent = np.random.normal(20 * np.expand_dims(sum_score / (sum_count + 1e-9), axis=1) / self.know_num,
                                        0.01, size=(self.time_window_num, self.stu_num, self.know_num))

        for iteration in range(epoch):
            u_latent_tmp, i_latent_tmp = np.copy(u_latent), np.copy(i_latent)
            alpha_tmp, B_tmp = np.copy(alpha), np.copy(B)
            i_gradient = np.zeros(shape=(self.prob_num, self.know_num))
            b_gradient = np.zeros(shape=self.prob_num)
            alpha_gradient = np.zeros(shape=self.stu_num)
            for t in range(self.time_window_num):
                u_gradient_t = np.zeros(shape=(self.stu_num, self.know_num))
                record_num_t = len(train_data[t])
                users = [record['user_id'] for record in train_data[t]]
                items = [record['item_id'] for record in train_data[t]]
                ratings = [record['score'] for record in train_data[t]]

                pred_R = [np.dot(u_latent[t][users[i]], i_latent[items[i]]) - B[items[i]] for i in range(record_num_t)]
                pred_u, freq_norm = stu_curve(u_latent, alpha, self.args.r, self.args.D, self.args.deltaT, self.args.S,
                                              time_freq)  # both shape are (time_window_num, stu_num, know_num)
                for i in range(record_num_t):
                    user, item, rating = users[i], items[i], ratings[i]
                    R_diff = pred_R[i] - rating
                    b_gradient[item] -= R_diff
                    u_gradient_t[user] += R_diff * i_latent[item]
                    i_gradient[item] += R_diff * u_latent[t][user] + self.args.lambda_V * i_latent[item]
                    i_gradient[item] -= mode_ind * self.args.lambda_V * np.sum(
                        np.expand_dims(self.exer_neigh[item], axis=1) * i_latent, axis=0) / sum(self.exer_neigh[item])
                    if t == 0:
                        u_gradient_t[user] += self.args.lambda_U_1 * u_latent[0][user]
                    else:
                        u_gradient_t[user] += self.args.lambda_U * (u_latent[t][user] - pred_u[t - 1][user])
                        alpha_gradient[user] += np.dot(pred_u[t - 1][user] - u_latent[t][user], u_latent[t][user] * (
                            freq_norm[t - 1][user] - np.exp(-self.args.deltaT / self.args.S)))
                    if t < self.time_window_num - 1:
                        u_gradient_t[user] += self.args.lambda_U * (pred_u[t][user] - u_latent[t + 1][user]) * (
                            alpha[user] * freq_norm[t][user] + (1 - alpha[user]) * np.exp(
                                - self.args.deltaT / self.args.S))
                    o1, o2 = np.where(self.par_mat[item] == 1)
                    for j in range(len(o1)):
                        i_gradient[item][o1[j]] -= self.args.lambda_P * 0.5 * (1 - np.tanh(
                            0.5 * (i_latent[item][o1[j]] - i_latent[item][o2[j]])))
                        i_gradient[item][o2[j]] += self.args.lambda_P * 0.5 * (1 - np.tanh(
                            0.5 * (i_latent[item][o1[j]] - i_latent[item][o2[j]])))
                u_latent[t] -= lr * u_gradient_t
            i_latent -= lr * i_gradient
            B -= lr_b * b_gradient
            alpha = np.clip(alpha - lr * alpha_gradient, 0, 1)
            change = max(np.max(np.abs(u_latent - u_latent_tmp)), np.max(np.abs(i_latent - i_latent_tmp)),
                         np.max(np.abs(alpha - alpha_tmp)), np.max(np.abs(B - B_tmp)))
            if iteration > 20 and change < epsilon:
                break
        self.u_latent, self.i_latent, self.alpha, self.B = u_latent, i_latent, alpha, B

    def eval(self, test_data) -> tuple:
        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            predict_rating = np.clip(np.dot(self.u_latent[-1][stu], self.i_latent[test_id]) - self.B[test_id], 0, 1)
            test_rmse.append((predict_rating - true_score) ** 2)
            test_mae.append(abs(predict_rating - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"U": self.u_latent, "V": self.i_latent, "alpha": self.alpha, "B": self.B}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.u_latent, self.i_latent, self.alpha, self.B = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)
