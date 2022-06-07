import xgboost as xgb
import numpy as np


class XGBoost():
    """
    XGBoost is an optimized distributed gradient boosting machine learning algorithm.

    Args:
        *n_estimators (int): Number of boosting iterations. Default: 10
        *max_depth (int): Maximum tree depth for base learners. Default: 5
        *verbosity (int): The degree of verbosity. Valid values are 0 (silent) - 3 (debug). Default: 0
        *objective (string or callable):
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used. Default: ``'reg:squarederror'``
        *eval_metric (str, list of str, or callable, optional):
            If a str, should be a built-in evaluation metric to use. See more in
            `API Reference of XGBoost Library <https://xgboost.readthedocs.io/en/latest/python/python_api.html>`_.
            Default: ``'rmse'``
    """
    def __init__(self, n_estimators=10, max_depth=5, verbosity=0, objective='reg:squarederror', eval_metric='rmse'):
        
        self.param = {
            'max_depth': max_depth,
            'verbosity': verbosity,
            'objective': objective,
            'eval_metric': eval_metric
        }
        self.n_estimators = n_estimators

    def fit(self, X, y):
        '''
        Training method. 

        Args:
            X(np.ndarray/scipy.sparse/pd.DataFrame/dt.Frame): The training input samples.
            y(np.ndarray, optional): The target values of training samples.
        '''
        train_matrix = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.param, train_matrix, self.n_estimators)

    def predict(self, X):
        '''
        Prediction method.
        
        :Returns: Predicted values with shape as [time_slot_num, node_num, 1].
        :Return type: np.ndarray
        '''
        test_matrix = xgb.DMatrix(X)
        return self.model.predict(test_matrix)


