import numpy as np

import warnings
warnings.filterwarnings("ignore")


class HM(object):
    '''
    Historical Mean. A naive method that simply return average of hisrory data of each time slot.

    Args:
        c(int): The number of time slots of closeness history. 
        p (int): The number of time slots of period history which presents daily feature.
        t (int): The number of time slots of trend history which presents weekly feature.
        Note that `(c, p, t)` cannot all be zero at the same time. They denote how many
        features should be considerd in average.
    '''
    def __init__(self, c, p, t):

        self.c = c
        self.p = p
        self.t = t

        if self.c == 0 and self.p == 0 and self.t == 0:
            raise ValueError('c p t cannot all be zero at the same time')

    def predict(self, closeness_feature, period_feature, trend_feature):
        '''
        Give closeness, period and trend history values and then use their averages as predict.
        '''
        prediction = []

        if self.c > 0:
            prediction.append(closeness_feature[:, :, :, 0])

        if self.p > 0:
            prediction.append(period_feature[:, :, :, 0])

        if self.t > 0:
            prediction.append(trend_feature[:, :, :, 0])

        prediction = np.mean(np.concatenate(prediction, axis=-1), axis=-1, keepdims=True)

        return prediction
