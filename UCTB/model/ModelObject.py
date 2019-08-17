import numpy as np

from ..dataset.data_loader import NodeTrafficLoader
from ..evaluation import metric


class ModelObject():
    """The base class of model.

    Attributes:
        test_y (np.ndarray): If ``make_test_data()`` has a :obj:`NodeTrafficLoader` as input, its ``test_y`` will be
            stored and used in ``eval()``.
        models (list): A list of models applied to each node.
        results (list or np.ndarray): A list of results on data of each node by ``predict()``. It can also be an
            ndarray with the shape of [node_num, test_time_slot_num].
    """
    def __init__(self):
        self.test_y = []
        self.models = []
        self.results = []

    @staticmethod
    def make_train_data_concat(X, y=None):
        """A method to make training data.

        Args:
            X (:obj:`NodeTrafficLoader` or np.ndarray): The training input samples.
                If it is an NodeTrafficLoader, closeness, period and trend history data will be concatenated.
                If it is an ndarray, its shape should be [time_slot_num, node_num, feature_num, 1].
            y (np.ndarray, optional): The target values of training samples.
                Its shape is [time_slot_num, node_num, 1]. It will be omitted if ``X`` is an NodeTrafficLoader.
                Default: ``None``

        Returns:
            tuple: (train_x, train_y)

            train_x (np.ndarray): The processed training input samples.
            If ``X`` is a ``NodeTrafficLoader``, function will concatenate it training closeness, period and trend
            history data. Its shape is [time_slot_num, node_num, feature_num, 1].

            train_y (np.ndarray): The processed target values with shape [time_slot_num, node_num, 1].
        """
        if type(X) is NodeTrafficLoader:
            return X.make_concat('all', True), X.train_y
        else:
            return X, y

    @staticmethod
    def make_train_data(X, y=None):
        """A method to make training data.

        Args:
            X (:obj:`NodeTrafficLoader` or list): The training input samples.
                If it is a list, it should includes three ndarrays as closeness, period and trend data,
                each either has a shape of [time_slot_num, node_num, feature_num, 1] or is an empty ndarray.
            y (np.ndarray, optional): The target values of training samples.
                Its shape is [time_slot_num, node_num, 1]. It will be omitted if ``X`` is an NodeTrafficLoader.
                Default: ``None``

        Returns:
            tuple: (train_x, train_y)

            train_x (tuple): The processed training input samples. A list of closeness, period and trend data,
            each is an ndarray with shape as [time_slot_num, node_num, feature_num, 1].

            train_y (np.ndarray): The processed target values with shape as [time_slot_num, node_num, 1].
        """
        if type(X) is NodeTrafficLoader:
            return (X.train_closeness, X.train_period, X.train_trend), X.train_y
        else:
            return tuple(X), y

    def make_test_data_concat(self, X):
        """A method to make test data.

        Args:
            X (:obj:`NodeTrafficLoader` or np.ndarray): The test input samples.
                If it is an ndarray, its shape should be [time_slot_num, node_num, feature_num, 1].
                If it is an NodeTrafficLoader, closeness, period and trend history data will be concatenated,
                and class will automatically set ``self.test_y`` as ``X.test_y``.

        Returns:
            np.ndarray: The processed test input samples with shape as [time_slot_num, node_num, feature_num, 1].
        """
        if type(X) is NodeTrafficLoader:
            self.test_y = X.test_y
            return X.make_concat('all', False)
        else:
            return X

    def make_test_data(self, X):
        """A method to make test data.

        Args:
            X (:obj:`NodeTrafficLoader` or list): The test input samples.
                If it is a list, it should includes three ndarrays as closeness, period and trend data,
                each either has a shape of [time_slot_num, node_num, feature_num, 1] or is an empty ndarray.
                If it is an NodeTrafficLoader, class will automatically set ``self.test_y`` as ``X.test_y``.

        Returns:
            tuple: The processed test input samples.
            A tuple of closeness, period and trend data,
            each is an ndarray with shape as [time_slot_num, node_num, feature_num, 1].
        """
        if type(X) is NodeTrafficLoader:
            self.test_y = X.test_y
            return X.test_closeness, X.test_period, X.test_trend
        else:
            return tuple(X)

    def fit(self, X, y=None):
        """Fitting method. Will be over ridden by derived class."""
        pass

    def predict(self, X):
        """Prediction method. Will be over ridden by derived class."""
        pass

    def eval(self, data=None):
        """Evaluatoin method.

        Args:
            data (np.ndarray, optional): The actual values of testing data with shape as [time_slot_num, node_num, 1].
                It can be omitted if object has stored data in ``self.test_y``. Default: ``None``

        Return:
            float: The RMSE (Root Mean Squared Error) of predicted results and actual values.
        """
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
        return rmse

