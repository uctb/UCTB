import xgboost as xgb
import numpy as np

from .ModelObject import ModelObject


class XGBoost(ModelObject):

    def __init__(self, n_estimators=10, max_depth=5, verbosity=0, objective='reg:squarederror', eval_metric='rmse'):
        """XGBoost is an optimized distributed gradient boosting machine learning algorithm.

        Args:
            n_estimators (int): Number of boosting iterations. Default: 10
            max_depth (int): Maximum tree depth for base learners. Default: 5
            verbosity (int): The degree of verbosity. Valid values are 0 (silent) - 3 (debug). Default: 0
            objective (string or callable):
                Specify the learning task and the corresponding learning objective or
                a custom objective function to be used. Default: ``'reg:squarederror'``
            eval_metric (str, list of str, or callable, optional):
                If a str, should be a built-in evaluation metric to use. See more in
                `API Reference of XGBoost Library <https://xgboost.readthedocs.io/en/latest/python/python_api.html>`_.
                Default: ``'rmse'``
        """
        super(XGBoost, self).__init__()
        self.param = {
            'max_depth': max_depth,
            'verbosity ': verbosity,
            'objective': objective,
            'eval_metric': eval_metric
        }
        self.n_estimators = n_estimators

    def fit(self, X, y=None):
        """Training method.

        Args:
            X (:obj:`NodeTrafficLoader` or np.ndarray): The training input samples.
                If it is an NodeTrafficLoader, closeness, period and trend history data will be concatenated in training.
                If it is an ndarray, its shape should be [time_slot_num, node_num, feature_num, 1].
            y (np.ndarray, optional): The target values of training samples.
                Its shape is [time_slot_num, node_num, 1]. It will be omitted if ``X`` is an NodeTrafficLoader.
                Default: ``None``

        Returns:
           list: It returns ``self.models``, which stores the models that are trained on each node's data.
        """
        self.models = []
        train_x, train_y = self.make_train_data_concat(X, y)
        node_num = train_x.shape[1]
        for i in range(node_num):
            train_matrix = xgb.DMatrix(train_x[:, i, :, -1], label=train_y[:, i])
            model = xgb.train(self.param, train_matrix, self.n_estimators)
            self.models.append(model)
        return self.models

    def predict(self, X):
        """Prediction method.

        Args:
            X (:obj:`NodeTrafficLoader` or np.ndarray): The test input samples.
                If it is an ndarray, its shape should be [time_slot_num, node_num, feature_num, 1].
                If it is an NodeTrafficLoader, method will concatenate closeness, period and trend history data
                in prediction.

        Returns:
            np.ndarray: Prediction results with shape as [time_slot_num, node_num, 1].
        """
        self.results = []
        test_x = self.make_test_data_concat(X)
        node_num = test_x.shape[1]
        for i in range(node_num):
            test_matrix = xgb.DMatrix(test_x[:, i, :, -1])
            result = self.models[i].predict(test_matrix)
            self.results.append(result)
        return np.expand_dims(np.stack(self.results, axis=1), 2)


