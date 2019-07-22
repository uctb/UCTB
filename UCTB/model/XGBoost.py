import xgboost as xgb
import numpy as np

from .ModelObject import ModelObject


class XGBoost(ModelObject):

    def __init__(self, n_estimators=10, max_depth=5, verbosity=0, objective='reg:squarederror', eval_metric='rmse'):
        super(XGBoost, self).__init__()
        self.param = {
            'max_depth': max_depth,
            'verbosity ': verbosity,
            'objective': objective,
            'eval_metric': eval_metric
        }
        self.n_estimators = n_estimators

    def fit(self, X, y=None):
        self.models = []
        train_x, train_y = self.make_train_data_concat(X, y)
        node_num = train_x.shape[1]
        for i in range(node_num):
            train_matrix = xgb.DMatrix(train_x[:, i, :, -1], label=train_y[:, i])
            model = xgb.train(self.param, train_matrix, self.n_estimators)
            self.models.append(model)
        return self.models

    def predict(self, X):
        self.results = []
        test_x = self.make_test_data_concat(X)
        node_num = test_x.shape[1]
        for i in range(node_num):
            test_matrix = xgb.DMatrix(test_x[:, i, :, -1])
            result = self.models[i].predict(test_matrix)
            self.results.append(result)
        return np.expand_dims(np.stack(self.results, axis=1), 2)


