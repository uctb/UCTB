from .ModelObject import ModelObject
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class GBRT(ModelObject):
    """Gradient Boosting for regression.

    Attributes:
        n_estimators (int): The number of boosting stages to perform. Default: 100
        max_depth (int): Maximum depth of the individual regression estimators.
            The maximum depth limits the number of nodes in the tree. Default: 5
    """
    def __init__(self, n_estimators=100, max_depth=5):
        super(GBRT, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y=None):
        """Training method.

        Args:
            X (:obj:`NodeTrafficLoader` or np.ndarray): The training input samples.
                If it is an NodeTrafficLoader, closeness, period and trend history data will be concatenated in training.
                If it is an ndarray, its shape should be [time_slot_num, node_num, feature_num, 1].
            y (np.ndarray, optional): The target values of training samples.
                Its shape is [time_slot_num, node_num, 1]. It will be omitted if ``X`` is an NodeTrafficLoader.
                Default: ``None``.

        Returns:
            list: It returns ``self.models``, which stores the models that are trained on each node's data.
        """
        self.models = []
        train_x, train_y = self.make_train_data_concat(X, y)
        node_num = train_x.shape[1]
        for i in range(node_num):
            model = GradientBoostingRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
            model.fit(train_x[:, i, :, -1], train_y[:, i])
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
            result = self.models[i].predict(test_x[:, i, :, -1])
            self.results.append(result)
        return np.expand_dims(np.stack(self.results, axis=1), 2)
