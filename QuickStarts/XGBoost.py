import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='DC', closeness_len=6, period_len=7, trend_len=4,
                                with_lm=False, normalize=False)

prediction_test = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(n_estimators=100, max_depth=3, objective='reg:squarederror')

    model.fit(np.concatenate((data_loader.train_closeness[:, i, :, 0],
                              data_loader.train_period[:, i, :, 0],
                              data_loader.train_trend[:, i, :, 0],), axis=-1),
              data_loader.train_y[:, i, 0])

    p_test = model.predict(np.concatenate((data_loader.test_closeness[:, i, :, 0],
                                           data_loader.test_period[:, i, :, 0],
                                           data_loader.test_trend[:, i, :, 0],), axis=-1))

    prediction_test.append(p_test.reshape([-1, 1, 1]))

prediction_test = np.concatenate(prediction_test, axis=-2)

print('Test RMSE', metric.rmse(prediction_test, data_loader.test_y, threshold=0))