import numpy as np
from UCTB.dataset import NodeTrafficLoader_CPT
from sklearn.ensemble import GradientBoostingRegressor
from UCTB.evaluation import metric

dataset = 'Bike'
city = 'DC'

data_loader = NodeTrafficLoader_CPT(dataset=dataset, city=city, with_lm=False,
                                    C_T=6, P_T=7, T_T=4)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = GradientBoostingRegressor(n_estimators=300, max_depth=5)

    train_x = np.concatenate([data_loader.train_closeness[:, 0, i, :],
                              data_loader.train_period[:, :, i, -1],
                              data_loader.train_trend[:, :, i, -1]], axis=-1)

    test_x = np.concatenate([data_loader.test_closeness[:, 0, i, :],
                             data_loader.test_period[:, :, i, -1],
                             data_loader.test_trend[:, :, i, -1]], axis=-1)

    model.fit(train_x, data_loader.train_y[:, i])

    p = model.predict(test_x).reshape([-1, 1, 1])

    prediction.append(p)

prediction = np.concatenate(prediction, axis=-2)

print(dataset, city, 'RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))