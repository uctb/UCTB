import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

class ARIMA(object):

    def __init__(self, train_data, order=None):

        self.train_data = train_data

        if order is None:
            # difference
            def stationary(series):
                t = self.adfTest(series)
                if t[0] < t[4]['1%']:
                    return True
                else:
                    return False
            order_i = 0
            while not stationary(np.diff(self.train_data, order_i)):
                order_i += 1
                if order_i > 2:
                    break
            self.order = sm.tsa.stattools.arma_order_select_ic(self.train_data,
                                                               max_ar=6, max_ma=2, ic=['aic']).aic_min_order
            self.order = list(self.order)
            self.order.insert(1, order_i)
        else:
            self.order = order

        print('ARIMA-Order:', self.order)

        self.model = sm.tsa.SARIMAX(train_data, order=self.order)
        self.model_res = self.model.fit(disp=False)

    def adfTest(self, timeSeries, maxlags=None, printFlag=True):
        t = sm.tsa.stattools.adfuller(timeSeries, maxlag=maxlags)
        if printFlag:
            output = pd.DataFrame(
                index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
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

    def predict(self, test_feature, minimum=0):
        result = []
        for i in range(len(test_feature)):
            model = sm.tsa.SARIMAX(test_feature[i], order=self.order)
            model_res = model.filter(self.model_res.params)
            p = model_res.forecast(1)
            result.append([p[0] if p > minimum else minimum])
        return np.array(result, dtype=np.float32)