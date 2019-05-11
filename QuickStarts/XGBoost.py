import numpy as np
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='DC', with_lm=False)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(max_depth=10)

    model.fit(data_loader.train_x[:, :, i, 0], data_loader.train_y[:, i], num_boost_round=20)

    p = model.predict(data_loader.test_x[:, :, i, 0]).reshape([-1, 1, 1])

    prediction.append(p)

prediction = np.concatenate(prediction, axis=-2)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))