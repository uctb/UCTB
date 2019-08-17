from .ModelObject import ModelObject

import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")


class ARIMA(ModelObject):
    """ARIMA (Autoregressive Integrated Moving Average)

    ARIMA is a generalization of an ARMA (Autoregressive Moving Average) model, used in predicting future points
    in time series analysis. Since there may be three kinds of series data as closeness, period and trend history,
    this class trains three different ARIMA models for each node according to the three kinds of history data,
    and returns average of the predicted values by the models in prediction.

    Args:
        order (tuple or ``None``): If set to a tuple of 3 integers, it indicates AR orders, intergration orders,
            and MA orders. If set to ``None``, class will calculate the orders for each series based on
            ``max_ar``, ``max_ma`` and ``max_d``. Default: ``None``
        max_ar (int): Maximum number of AR lags to use. Default: 6
        max_ma (int): Maximum number of MA lags to use. Default: 2
        max_d (int): Maximum number of degrees of differencing. Default: 2
        forecast_step (int): The number of predicted future steps. Default: 1

    Attributes:
        orders (list): It stores the (p, d, q) orders corresponding to each model stored in ``self.models``.
    """
    def __init__(self, order=None, max_ar=6, max_ma=2, max_d=2, forecast_step=1):
        super(ARIMA, self).__init__()

        self.order = order
        self.max_ar = max_ar
        self.max_ma = max_ma
        self.max_d = max_d
        self.forecast_step = forecast_step

        self.orders = []

    @staticmethod
    def get_order(series, order=None, max_ar=6, max_ma=2, max_d=2):
        """If ``order`` is ``None``, it simply returns ``order``,
        otherwise, it calculates the (p, d, q) orders for the series data
        based on ``max_ar``, ``max_ma`` and ``max_d``.

        Attributes:
            series (np.ndarray): A series of data with shape as [time_slot_num].
            order (tuple or ``None'``): If type is tuple, function will return it.
                If it is ``'None``, function will calculate the orders. Default: ``None``
            max_ar (int): Maximum number of AR lags to use. Default: 6
            max_ma (int): Maximum number of MA lags to use. Default: 2
            max_d (int): Maximum number of degrees of differencing. Default: 2

        Returns:
            tuple: A tuple of 3 integers. The (p, d, q) orders for the series data.
        """
        if order is None:
            # difference
            def stationary(series):
                t = ARIMA.adf_test(series)
                if t[0] < t[4]['1%']:
                    return True
                else:
                    return False
            order_i = 0
            while not stationary(np.diff(series, order_i)):
                order_i += 1
                if order_i > max_d:
                    break
            order = sm.tsa.stattools.arma_order_select_ic(np.diff(series, order_i),
                                                          max_ar=max_ar, max_ma=max_ma,
                                                          ic=['aic']).aic_min_order
            order = list(order)
            order.insert(1, order_i)
        return order

    @staticmethod
    def adf_test(time_series, max_lags=None, verbose=False):
        """Augmented Dickey–Fuller test.

        The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process
        in the presence of serial correlation.
        """
        t = sm.tsa.stattools.adfuller(time_series, maxlag=max_lags)
        if verbose:
            output = pd.DataFrame(
                index=['Test Statistic Value',
                       "p-value",
                       "Lags Used",
                       "Number of Observations Used",
                       "Critical Value(1%)",
                       "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
            output['value']['Test Statistic Value'] = t[0]
            output['value']['p-value'] = t[1]
            output['value']['Lags Used'] = t[2]
            output['value']['Number of Observations Used'] = t[3]
            output['value']['Critical Value(1%)'] = t[4]['1%']
            output['value']['Critical Value(5%)'] = t[4]['5%']
            output['value']['Critical Value(10%)'] = t[4]['10%']
            print(output)
        return t

    def _fit(self, data):
        """The inner fitting method.

        Args:
            data (np.ndarray): A series of data with shape as [time_slot_num].

        Returns:
            tuple: (order, model_res)

            order (tuple): The (p, d, q) orders of ARIMA model on data.

            model_res (Results Object): An object that stores parameters of model.
                It will be ``None`` if errors occurred during fitting.
        """
        order = self.get_order(data, self.order)
        model = sm.tsa.SARIMAX(data, order=order, max_ar=self.max_ar, max_ma=self.max_ma, max_d=self.max_d)

        try:
            model_res = model.fit(disp=False)
        except:
            model_res = None

        return order, model_res

    def fit(self, X, y=None):
        """Fitting method.

        Args:
            X (:obj:`NodeTrafficLoader` or list): The training input samples.
                If it is a list, it should includes three ndarrays as closeness, period and trend data,
                each either has a shape of [time_slot_num, node_num, feature_num, 1] or is an empty ndarray.
            y (np.ndarray, optional): The target values of training samples.
                Its shape is [time_slot_num, node_num, 1]. It will be omitted if ``X`` is an NodeTrafficLoader.
                Default: ``None``

        Returns:
            tuple: (models, orders)

            models (list): ``self.models``. Models that are trained on each node's data.

            orders (list): ``self.orders``. The (p, d, q) orders of models that are trained on each node's data.
        """

        self.models = []
        self.orders = []

        train_x, train_y = self.make_train_data(X, y)
        closeness_feature, period_feature, trend_feature = train_x
        node_num = train_y.shape[1]

        for i in range(node_num):
            self.models.append([None] * 3)
            self.orders.append([None] * 3)
            if closeness_feature is not None and 0 not in closeness_feature.shape:
                data = closeness_feature[:, i, -1, 0]
                order, model_res = self._fit(data)
                self.models[-1][0] = model_res
                self.orders[-1][0] = order

            if period_feature is not None and 0 not in period_feature.shape:
                data = period_feature[:, i, -1, 0]
                order, model_res = self._fit(data)
                self.models[-1][1] = model_res
                self.orders[-1][1] = order

            if trend_feature is not None and 0 not in trend_feature.shape:
                data = trend_feature[:, i, -1, 0]
                order, model_res = self._fit(data)
                self.models[-1][2] = model_res
                self.orders[-1][2] = order

        return self.models, self.orders

    @staticmethod
    def _predict(model_res, order, X, forecast_step):
        """The inner prediction method.

        Args:
            model_res (Results Object): The object of a model as ``_fit`` returns.
            order (tuple): The (p, d, q) orders of the model.
            X (np.ndarray): The test series data with shape as [time_slot_num].
            forecast_step (int): The number of predicted future steps.

        Returns:
            float: The prediction result.
        """
        model = sm.tsa.SARIMAX(X, order=order)

        try:
            model_res = model.filter(model_res.params)
            p = model_res.forecast(forecast_step)[-1]
        except AttributeError:
            p = 0

        return p

    def predict(self, X):
        """Prediction Method.

        Args:
            X (:obj:`NodeTrafficLoader` or list): The test input samples.
                If it is a list, it should includes three ndarrays as closeness, period and trend data,
                each either has a shape of [time_slot_num, node_num, feature_num, 1] or is an empty ndarray.

        Returns:
            np.ndarray: Prediction results with shape as [time_slot_num, node_num, 1].
        """
        closeness_feature, period_feature, trend_feature = self.make_test_data(X)
        slot_num = 0
        node_num = 0
        feature_num = 0
        if closeness_feature is not None and 0 not in closeness_feature.shape:
            slot_num = closeness_feature.shape[0]
            node_num = closeness_feature.shape[1]
            feature_num += 1
        if period_feature is not None and 0 not in period_feature.shape:
            slot_num = period_feature.shape[0]
            node_num = period_feature.shape[1]
            feature_num += 1
        if trend_feature is not None and 0 not in trend_feature.shape:
            slot_num = trend_feature.shape[0]
            node_num = trend_feature.shape[1]
            feature_num += 1
        self.results = np.zeros((node_num, slot_num), dtype=np.float32)

        for i in range(node_num):
            for j in range(slot_num):
                if closeness_feature is not None and 0 not in closeness_feature.shape:
                    self.results[i][j] += self._predict(self.models[i][0], self.orders[i][0],
                                                        closeness_feature[j, i, :, 0], self.forecast_step)
                if period_feature is not None and 0 not in period_feature.shape:
                    self.results[i][j] += self._predict(self.models[i][1], self.orders[i][1],
                                                        period_feature[j, i, :, 0], self.forecast_step)
                if trend_feature is not None and 0 not in trend_feature.shape:
                    self.results[i][j] += self._predict(self.models[i][2], self.orders[i][2],
                                                        trend_feature[j, i, :, 0], self.forecast_step)

        self.results /= feature_num
        return np.expand_dims(np.transpose(self.results, (1, 0)), 2)
