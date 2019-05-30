import numpy as np
from UCTB.dataset import NodeTrafficLoader_CPT, NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader_CPT(dataset='Metro', city='Chongqing', with_lm=False,
                                    test_ratio=0.1, normalize=False,
                                    C_T=5, P_T=4, T_T=3)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(max_depth=4)

    train_x = np.concatenate([data_loader.train_closeness[:, 0, i, :],
                              data_loader.train_period[:, :, i, -1],
                              data_loader.train_trend[:, :, i, -1]], axis=-1)

    test_x = np.concatenate([data_loader.test_closeness[:, 0, i, :],
                             data_loader.test_period[:, :, i, -1],
                             data_loader.test_trend[:, :, i, -1]], axis=-1)

    model.fit(train_x, data_loader.train_y[:, i], num_boost_round=159)

    p = model.predict(test_x).reshape([-1, 1, 1])

    prediction.append(p)

prediction = np.concatenate(prediction, axis=-2)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))