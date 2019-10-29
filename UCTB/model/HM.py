import numpy as np

import warnings
warnings.filterwarnings("ignore")


class HM(object):

    def __init__(self, c, p, t):

        self.c = c
        self.p = p
        self.t = t

        if self.c == 0 and self.p == 0 and self.t == 0:
            raise ValueError('c p t cannot all be zero at the same time')

    def predict(self, closeness_feature, period_feature, trend_feature):

        prediction = []

        if self.c > 0:
            prediction.append(closeness_feature[:, :, :, 0])

        if self.p > 0:
            prediction.append(period_feature[:, :, :, 0])

        if self.t > 0:
            prediction.append(trend_feature[:, :, :, 0])

        prediction = np.mean(np.concatenate(prediction, axis=-1), axis=-1, keepdims=True)

        return prediction
