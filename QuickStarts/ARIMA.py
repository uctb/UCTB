import numpy as np

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='DiDi', city='Chengdu',
                                closeness_len=6, period_len=7, trend_len=4,
                                with_lm=False, with_tpe=False)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    try:
        model_obj = ARIMA(closeness_feature=data_loader.train_closeness[:, i, -1, 0],
                          period_feature=data_loader.train_period[:, i, -1, 0],
                          trend_feature=data_loader.train_period[:, i, -1, 0],
                          order=(6, 0, 2))
                          # max_ar=10, max_ma=5, max_d=2)

        p = model_obj.predict(closeness_feature=data_loader.test_closeness[:, i, :, 0],
                              period_feature=data_loader.test_period[:, i, :, 0],
                              trend_feature=data_loader.test_trend[:, i, :, 0], forecast_step=1)

    except Exception as e:
        print('Converge failed with error', e)
        print('Using zero as prediction')
        p = np.zeros([data_loader.test_y.shape[0], 1, 1])

    prediction.append(p)

prediction = np.concatenate(prediction, axis=-2)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))