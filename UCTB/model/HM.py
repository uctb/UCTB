from ..dataset import NodeTrafficLoader
from .ModelObject import ModelObject

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class HM(ModelObject):

    def __init__(self, c, p, t):

        super(HM, self).__init__()
        self.c = c
        self.p = p
        self.t = t

        if self.c == 0 and self.p == 0 and self.t == 0:
            raise ValueError('c p t cannot all be zero at the same time')

    def fit(self, X=None, y=None):
        print("HM doesn't need fitting.")

    def predict(self, X):

        self.results = []

        closeness_feature, period_feature, trend_feature = self.make_test_data(X)
        if self.c > 0:
            assert self.c <= closeness_feature.shape[2]
        if self.p > 0:
            assert self.p <= period_feature.shape[2]
        if self.t > 0:
            assert self.t <= trend_feature.shape[2]

        prediction = []

        if self.c > 0:
            prediction.append(closeness_feature[:, :, -self.c:, 0])

        if self.p > 0:
            prediction.append(period_feature[:, :, -self.p:, 0])

        if self.t > 0:
            prediction.append(trend_feature[:, :, -self.t:, 0])

        prediction = np.mean(np.concatenate(prediction, axis=-1), axis=-1, keepdims=True)
        for i in range(prediction.shape[1]):
            self.results.append(prediction[:, i])

        return prediction
