from .ModelObject import ModelObject

import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

class ARIMA(ModelObject):

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
    def adf_test(time_series, max_lags=None, verbose=True):
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
        order = self.get_order(data, self.order)
        model = sm.tsa.SARIMAX(data, order=order, max_ar=self.max_ar, max_ma=self.max_ma, max_d=self.max_d)

        try:
            model_res = model.fit(disp=False)
        except:
            model_res = None

        return order, model_res

    def fit(self, X, y=None):

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
        model = sm.tsa.SARIMAX(X, order=order)

        try:
            model_res = model.filter(model_res.params)
            p = model_res.forecast(forecast_step)[-1]
        except AttributeError:
            p = 0

        return p

    def predict(self, X):

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

class ARIMA2(object):

    def __init__(self, closeness_feature, period_feature, trend_feature,
                 order=None, max_ar=6, max_ma=2, max_d=2):

        self.temporal_features = []
        if closeness_feature is not None and 0 not in closeness_feature.shape:
            self.temporal_features.append(closeness_feature)
        if period_feature is not None and 0 not in period_feature.shape:
            self.temporal_features.append(period_feature)
        if trend_feature is not None and 0 not in trend_feature.shape:
            self.temporal_features.append(trend_feature)

        assert len(self.temporal_features) > 0

        self.order = []
        self.model_res = []
        for data in self.temporal_features:
            order = self.get_order(data, order)
            model = sm.tsa.SARIMAX(data, order=order, max_ar=max_ar, max_ma=max_ma, max_d=max_d)
            model_res = model.fit(disp=False)
            self.order.append(order)
            self.model_res.append(model_res)

    def get_order(self, series, order=None, max_ar=6, max_ma=2, max_d=2):
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
    def adf_test(time_series, max_lags=None, verbose=True):
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

    def predict(self, closeness_feature, period_feature, trend_feature, forecast_step=1):

        temporal_features = []
        if closeness_feature is not None and 0 not in closeness_feature.shape:
            temporal_features.append(closeness_feature)
        if period_feature is not None and 0 not in period_feature.shape:
            temporal_features.append(period_feature)
        if trend_feature is not None and 0 not in trend_feature.shape:
            temporal_features.append(trend_feature)

        assert len(temporal_features) == len(self.temporal_features)

        result = []
        for index in range(len(temporal_features)):
            tmp_result = []
            for i in range(len(temporal_features[index])):
                model = sm.tsa.SARIMAX(temporal_features[index][i], order=self.order[index])
                model_res = model.filter(self.model_res[index].params)
                p = model_res.forecast(forecast_step).reshape([1, -1])
                tmp_result.append(p)
            result.append(np.array(tmp_result, dtype=np.float32))
        return np.mean(result, dtype=np.float32, axis=0)