from .ModelObject import ModelObject
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class GBRT(ModelObject):
    def __init__(self, n_estimators=10, max_depth=5):
        super(GBRT, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y=None):
        self.models = []
        train_x, train_y = self.make_train_data_concat(X, y)
        node_num = train_x.shape[1]
        for i in range(node_num):
            model = GradientBoostingRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
            model.fit(train_x[:, i, :, -1], train_y[:, i])
            self.models.append(model)
        return self.models

    def predict(self, X):
        self.results = []
        test_x = self.make_test_data_concat(X)
        node_num = test_x.shape[1]
        for i in range(node_num):
            result = self.models[i].predict(test_x[:, i, :, -1])
            self.results.append(result)
        return np.expand_dims(np.stack(self.results, axis=1), 2)
