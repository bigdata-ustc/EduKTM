# Code reused from https://github.com/arghosh/AKT.git


import numpy as np
import math

class Data(object):
    def __init__(self, n_question, seqlen, separate_char):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self, path):
        f_data = open(path , 'r')
        q_data = []
        qa_data = []
        for line_id, line in enumerate(f_data):
            line = line.strip( )
            # line_id starts from 0
            if line_id % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                #print(len(Q))
            elif line_id % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]
                #print(len(A),A)

                # start split the data
                n_split = 1
                #print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                #print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        end_index  = len(A)
                    else:
                        end_index = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, end_index):
                        if len(Q[i]) > 0 :
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_data_array = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_data_array[j, :len(dat)] = dat

        qa_data_array = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_data_array[j, :len(dat)] = dat
        # data_array: [ array([[],[],..])] Shape: (3633, 200)
        return q_data_array, qa_data_array
