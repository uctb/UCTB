from ..dataset import NodeTrafficLoader
from .ModelObject import ModelObject

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class HM(ModelObject):
    """Historical Mean. A naive method that simply return average of hisrory data of each time slot.

    Args:
        c (int): The number of time slots of closeness history of one data sample. Default: 0
        p (int): The number of time slots of period history of one data sample. Default: 0
        t (int): The number of time slots of trend history of one data sample. Default: 4
    """
    def __init__(self, c=0, p=0, t=4):

        super(HM, self).__init__()
        self.c = c
        self.p = p
        self.t = t

        if self.c == 0 and self.p == 0 and self.t == 0:
            raise ValueError('c p t cannot all be zero at the same time')

    def fit(self, X=None, y=None):
        """HM doesn't need fitting."""
        print("HM doesn't need fitting.")

    def predict(self, X):
        """Prediction method.

        Args:
            X (:obj:`Dataloader` or list): The test input samples.
                If it is a list, it should includes three ndarrays as closeness, period and trend data,
                each either has a shape of [time_slot_num, node_num, feature_num, 1] or is an empty ndarray.

        Raises:
            AssertionError: If ``X`` has shorter history data as c/p/t required.

        Returns:
            np.ndarray: Results with shape as [time_slot_num, node_num, 1].
        """
        self.results = []

        closeness_feature, period_feature, trend_feature = self.make_test_data(X)
        assert self.c == 0 or self.c <= closeness_feature.shape[2]
        assert self.p == 0 or self.p <= period_feature.shape[2]
        assert self.t == 0 or self.t <= trend_feature.shape[2]

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
