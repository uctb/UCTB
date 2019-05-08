import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

class HM(object):

    def __init__(self, c, p, t):

        self.c = c
        self.p = p
        self.t = t

    def predict(self, start_index, sequence_Data, time_fitness):

        '''
        :param time_fitness: int, minutes
        '''

        prediction = []

        for i in range(start_index, len(sequence_Data)):

            p = []

            for j in range(1, self.t + 1):

                p.append(sequence_Data[i - j * int(7 * 24 * 60 / time_fitness)])

            for j in range(1, self.p + 1):

                p.append(sequence_Data[i - j * int(24 * 60 / time_fitness)])

            for j in range(1, self.c + 1):

                p.append(sequence_Data[i - int(j * 60 / time_fitness)])

            prediction.append(np.mean(p, axis=0, keepdims=True))

        prediction = np.concatenate(prediction, axis=0)

        return prediction
