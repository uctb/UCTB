import numpy as np

from ..dataset.data_loader import NodeTrafficLoader
from ..evaluation import metric


class ModelObject():

    def __init__(self):
        self.test_y = []
        self.model = None
        self.models = []
        self.results = []

    @staticmethod
    def make_train_data_concat(X, y=None):

        if type(X) is NodeTrafficLoader:
            return X.make_concat('all', True), X.train_y
        else:
            return X, y

    @staticmethod
    def make_train_data(X, y=None):

        if type(X) is NodeTrafficLoader:
            return (X.train_closeness, X.train_period, X.train_trend), X.train_y
        else:
            return X, y

    def make_test_data_concat(self, X):

        if type(X) is NodeTrafficLoader:
            self.test_y = X.test_y
            return X.make_concat('all', False)
        else:
            return X

    def make_test_data(self, X):
        if type(X) is NodeTrafficLoader:
            self.test_y = X.test_y
            return X.test_closeness, X.test_period, X.test_trend
        else:
            return X

    def eval(self, data=None):
        test_y = self.test_y
        if type(data) is NodeTrafficLoader:
            test_y = data.test_y
        elif type(data) is np.ndarray:
            test_y = data
        assert len(test_y) > 0
        results = self.results
        if type(results) is np.ndarray:
            results = np.transpose(results, (1, 0))
        if type(results) is list:
            results = np.stack(results, axis=1)
        if len(results.shape) == 2:
            results = np.expand_dims(results, 2)
        if len(test_y.shape) == 2:
            test_y = np.expand_dims(test_y, 2)
        assert test_y.shape == results.shape
        rmse = metric.rmse(results, test_y, threshold=0)
        print("RMSE", rmse)

