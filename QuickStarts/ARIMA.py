import numpy as np

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric


data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=24, period_len=0, trend_len=0,
                                with_lm=False, normalize=False)

test_rmse_collector = []

for i in range(data_loader.station_number):

    try:
        model_obj = ARIMA(time_sequence=data_loader.train_closeness[:, i, -1, 0],
                          order=[6, 0, 1], seasonal_order=[0, 0, 0, 0])

        test_prediction = model_obj.predict(time_sequences=data_loader.test_closeness[:, i, :, 0],
                                            forecast_step=1)

    except Exception as e:
        print('Converge failed with error', e)
        print('Using last as prediction')

        test_prediction = data_loader.test_closeness[:, i, -1:, :]

    test_rmse_collector.append(test_prediction)

    print('Station', i, metric.rmse(test_prediction, data_loader.test_y[:, i:i+1], threshold=0))

test_rmse_collector = np.concatenate(test_rmse_collector, axis=-2)
test_rmse = metric.rmse(test_rmse_collector, data_loader.test_y, threshold=0)

print('test_rmse', test_rmse)
