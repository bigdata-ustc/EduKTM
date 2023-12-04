# Code reused from https://github.com/arghosh/AKT.git
import numpy as np
import math


class DATA(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    time_factor sequence
    attempt factor sequence
    hint factor sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        a_data = []
        e_data = []
        time_data = []
        attempt_data = []
        hint_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 7 != 0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]

            if lineID % 7 == 2:
                A = line_data
            elif lineID % 7 == 3:
                E = line_data
            elif lineID % 7 == 4:
                Time = line_data
            elif lineID % 7 == 5:
                Attempt = line_data
            elif lineID % 7 == 6:
                Hint = line_data

                # start split the data after getting the final feature
                n_split = 1
                total_len = len(A)
                if total_len > self.seqlen:
                    n_split = math.floor(len(A) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    answer_sequence = []
                    exercise_sequence = []
                    time_sequence = []
                    attempt_sequence = []
                    hint_sequence = []
                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than 2
                    if end_index - k * self.seqlen > 2:
                        for i in range(k * self.seqlen, end_index):
                            answer_sequence.append(int(A[i]))
                            exercise_sequence.append(int(E[i]))
                            time_sequence.append(float(Time[i]))
                            attempt_sequence.append(float(Attempt[i]))
                            hint_sequence.append(float(Hint[i]))

                        # print('instance:-->', len(instance),instance)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        time_data.append(time_sequence)
                        attempt_data.append(attempt_sequence)
                        hint_data.append(hint_sequence)
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        time_dataArray = np.zeros((len(time_data), self.seqlen))
        for j in range(len(time_data)):
            dat = time_data[j]
            time_dataArray[j, :len(dat)] = dat

        attempt_dataArray = np.zeros((len(attempt_data), self.seqlen))
        for j in range(len(attempt_data)):
            dat = attempt_data[j]
            attempt_dataArray[j, :len(dat)] = dat

        hint_dataArray = np.zeros((len(hint_data), self.seqlen))
        for j in range(len(hint_data)):
            dat = hint_data[j]
            hint_dataArray[j, :len(dat)] = dat

        return e_dataArray, a_dataArray, time_dataArray, \
            attempt_dataArray, hint_dataArray
