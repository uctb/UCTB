import numpy as np
import statsmodels.api as sm


class ARIMA(object):

    def __init__(self, train_data, order):
        self.order = order
        self.model = sm.tsa.SARIMAX(train_data, order=self.order)
        self.model_res = self.model.fit(disp=False)

    def predict(self, test_feature, minimum=0):
        result = []
        for i in range(len(test_feature)):
            model = sm.tsa.SARIMAX(test_feature[i], order=self.order)
            model_res = model.filter(self.model_res.params)
            p = model_res.forecast(1)
            result.append(p if p > minimum else minimum)
        return np.array(result, dtype=np.float32)