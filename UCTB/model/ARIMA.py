import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")


class ARIMA(object):

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