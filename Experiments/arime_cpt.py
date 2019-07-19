import numpy as np

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

closeness_len = 6
period_len = 7
trend_len = 4

data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian',
                                closeness_len=closeness_len,
                                period_len=period_len,
                                trend_len=trend_len,
                                with_lm=False, with_tpe=False)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    # Build closeness predict model
    try:
        closeness_model = ARIMA(data_loader.train_closeness[:, i, -1, 0], [6, 0, 2])
        closeness_prediction = closeness_model.predict(data_loader.test_closeness[:, i, :, 0])
    except:
        # Converge failed with error, Using zero as prediction
        closeness_prediction = np.zeros([data_loader.test_closeness[:, i, :, 0].shape[0], 1, 1])

    # Build period predict model
    try:
        period_model = ARIMA(data_loader.train_period[:, i, -1, 0], [6, 0, 2])
        period_prediction = period_model.predict(data_loader.test_period[:, i, :, 0])
    except:
        # Converge failed with error, Using zero as prediction
        period_prediction = np.zeros([data_loader.test_period[:, i, :, 0].shape[0], 1, 1])

    # Build trend predict model
    try:
        trend_model = ARIMA(data_loader.train_trend[:, i, -1, 0], [6, 0, 2])
        trend_prediction = trend_model.predict(data_loader.test_trend[:, i, :, 0])
    except:
        # Converge failed with error, Using zero as prediction
        trend_prediction = np.zeros([data_loader.test_trend[:, i, :, 0].shape[0], 1, 1])

    prediction.append(np.mean([closeness_prediction, period_prediction, trend_prediction], axis=0))

    print(np.concatenate(prediction, axis=-2).shape)

prediction = np.concatenate(prediction, axis=-2)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))