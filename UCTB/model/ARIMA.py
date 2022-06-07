import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")


class ARIMA(object):
    """ARIMA is a generalization of an ARMA (Autoregressive Moving Average) model, used in predicting 
    future points in time series analysis.

    Since there may be three kinds of series data as closeness, period and trend history, this class
    trains three different ARIMA models for each node according  to the three kinds of history data, 
    and returns average of the predicted values by the models in prediction.

    Args:
        time_sequence(array_like): The observation value of time_series.
        order(iterable): It stores the (p, d, q) orders of the model for the number of AR parameters
            , differences, MA parameters. If set to None, ARIMA class will calculate the orders for 
            each series based on max_ar, max_ma and max_d. Default: None
        seasonal_order(iterable): It stores the (P,D,Q,s) order of the seasonal ARIMA model for the
            AR parameters, differences, MA parameters, and periodicity. `s` is an integer giving the 
            periodicity (number of periods in season).
        max_ar(int): Maximum number of AR lags to use. Default: 6
        max_ma(int): Maximum number of MA lags to use. Default: 4
        max_d(int): Maximum number of degrees of differencing. Default: 2

    Attribute:
        order(iterable): (p, d, q) orders for ARIMA model. 
        seasonal_order(iterable): (P,D,Q,s) order for seasonal ARIMA model. 
        model_res(): Fit method for likelihood based models. 
    """

    def __init__(self, time_sequence, order=None, seasonal_order=(0, 0, 0, 0), max_ar=6, max_ma=4, max_d=2):

        self.seasonal_order = seasonal_order
        auto_order = self.get_order(time_sequence, order, max_ar=max_ar, max_ma=max_ma, max_d=max_d)
        model = sm.tsa.SARIMAX(time_sequence, order=auto_order, seasonal_order=self.seasonal_order)
        model_res = model.fit(disp=False)
        self.order = auto_order
        self.model_res = model_res

    def get_order(self, series, order=None, max_ar=6, max_ma=2, max_d=2):
        '''
        If order is None, it simply returns order, otherwise, it calculates the (p, d, q) orders 
        for the series data based on max_ar, max_ma and max_d.
        '''

        def stationary(series):
            t = ARIMA.adf_test(series, verbose=False)
            if t[0] < t[4]['1%']:
                return True
            else:
                return False

        if order is None:
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
        '''
        Augmented Dickey–Fuller test. The Augmented Dickey-Fuller test can be used to test for 
        a unit root in a univariate process in the presence of serial correlation.
        '''
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

    def predict(self, time_sequences, forecast_step=1):
        '''
        Argues:
            time_sequences: The input time_series features.
            forecast_step: The number of predicted future steps. Default: 1
        
        :return: Prediction results with shape of (len(time_sequence)/forecast_step,forecast_step=,1).
        :type: np.ndarray
        '''
        result = []
        """ origin predict method, output shape is [math.ceil(len(time_sequences) / forecast_step), forecast_step]
        for i in range(0, len(time_sequences), forecast_step):
            fs = forecast_step if ((i + forecast_step) < len(time_sequences)) else (len(time_sequences) - i)
            model = sm.tsa.SARIMAX(time_sequences[i], order=self.order, seasonal_order=self.seasonal_order)
            model_res = model.filter(self.model_res.params)
            p = model_res.forecast(fs).reshape([-1, 1])
            result.append(p)
        """
        # new predict method, output shape is [len(time_sequences), forecast_step]
        for i in range(len(time_sequences)):
            model = sm.tsa.SARIMAX(time_sequences[i], order=self.order, seasonal_order=self.seasonal_order)
            model_res = model.filter(self.model_res.params)
            p = model_res.forecast(forecast_step)
            p = p.reshape([-1, forecast_step])
            result.append(p)
        if forecast_step != 1:
            result = np.concatenate(result, axis=0)
        return np.array(result, dtype=np.float32)