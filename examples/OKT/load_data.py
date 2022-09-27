import numpy as np
import math


class DATA(object):
    def __init__(self, seqlen, separate_char=',', has_at=True, selection_keys=None):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.has_at = has_at
        self.selection_keys = selection_keys

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    it sequence
    at sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        a_data = []
        e_data = []
        it_data = []
        at_data = []
        if self.has_at:
            n_unit = 6
        else:
            n_unit = 5
        for lineID, line in enumerate(f_data):
            line = line.strip()
            line_data = line.split(self.separate_char)
            if len(line_data[len(line_data) - 1]) == 0:
                line_data = line_data[:-1]

            if lineID % n_unit == 1:
                Q = line_data
            elif lineID % n_unit == 2:
                A = line_data
            elif lineID % n_unit == 3:
                E = line_data
            elif lineID % n_unit == 4:
                IT = line_data
            elif lineID % n_unit == 5:
                AT = line_data

            if lineID % n_unit == n_unit - 1:
                # start split the data
                n_split = 1
                total_len = len(A)
                if total_len > self.seqlen:
                    n_split = math.floor(len(A) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    exercise_sequence = []
                    it_sequence = []
                    at_sequence = []
                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than 2
                    if end_index - k * self.seqlen > 2:
                        for i in range(k * self.seqlen, end_index):
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(int(A[i]))
                            exercise_sequence.append(int(E[i]))
                            it_sequence.append(int(IT[i]))
                            if self.has_at:
                                at_sequence.append(int(AT[i]))

                        q_data.append(question_sequence)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        it_data.append(it_sequence)
                        at_data.append(at_sequence)
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        it_dataArray = np.zeros((len(it_data), self.seqlen))
        for j in range(len(it_data)):
            dat = it_data[j]
            it_dataArray[j, :len(dat)] = dat

        at_dataArray = np.zeros((len(at_data), self.seqlen))
        for j in range(len(at_data)):
            dat = at_data[j]
            at_dataArray[j, :len(dat)] = dat

        selection = {
            'q': q_dataArray,
            'a': a_dataArray,
            'e': e_dataArray,
            'it': it_dataArray,
            'at': at_dataArray,
        }
        res = []
        if self.selection_keys is None:
            res = selection.values()
        else:
            for k in self.selection_keys:
                res.append(selection[k])
        return tuple(res)
