import numpy as np

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='ChargeStation', city='Beijing')

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    try:
        model_obj = ARIMA(data_loader.train_data[:, i], [30, 0, 2])
        p = model_obj.predict(data_loader.test_x[:, :, i, 0])
    except Exception as e:
        print('Converge failed with error', e)
        print('Using zero as prediction')
        p = np.zeros([data_loader.test_x[:, :, i, 0].shape[0], 1, 1])

    prediction.append(p)

    print(np.concatenate(prediction, axis=-1).shape)

prediction = np.concatenate(prediction, axis=-1)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))