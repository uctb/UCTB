import nni
import numpy as np

from UCTB.dataset import NodeTrafficLoader_CPT, NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric

params = {
    'Dataset': 'Bike',
    'City': 'NYC',
    'CT': 6,
    'PT': 7,
    'TT': 0,
    'max_depth': 10,
    'num_boost_round': 150
}

params.update(nni.get_next_parameter())

data_loader = NodeTrafficLoader_CPT(dataset=params['Dataset'], city=params['City'],
                                    with_lm=False, test_ratio=0.1, normalize=False,
                                    C_T=int(params['CT']), P_T=int(params['PT']), T_T=int(params['TT']))

test_prediction = []
val_prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(max_depth=int(params['max_depth']))

    train = []
    test_x = []

    if int(params['CT']) > 0:
        train.append(data_loader.train_closeness[:, 0, i, :])
        test_x.append(data_loader.test_closeness[:, 0, i, :])
    if int(params['PT']) > 0:
        train.append(data_loader.train_period[:, :, i, -1])
        test_x.append(data_loader.test_period[:, :, i, -1])
    if int(params['TT']) > 0:
        train.append(data_loader.train_trend[:, :, i, -1])
        test_x.append(data_loader.test_trend[:, :, i, -1])

    train = np.concatenate(train, axis=-1)

    test_x = np.concatenate(test_x, axis=-1)

    # val has the same length as test
    train_x, val_x = train[:-len(test_x)], train[-len(test_x):]

    train_y, val_y = data_loader.train_y[:-len(test_x), i], data_loader.train_y[-len(test_x):, i]

    model.fit(train_x, train_y, num_boost_round=int(params['num_boost_round']))

    test_p = model.predict(test_x).reshape([-1, 1, 1])
    val_p = model.predict(val_x).reshape([-1, 1, 1])

    test_prediction.append(test_p)
    val_prediction.append(val_p)

test_prediction = np.concatenate(test_prediction, axis=-2)
val_prediction = np.concatenate(val_prediction, axis=-2)

val_rmse = metric.rmse(val_prediction, data_loader.train_y[-len(data_loader.test_y):], threshold=0)
test_rmse = metric.rmse(test_prediction, data_loader.test_y, threshold=0)

nni.report_final_result({
    'default': val_rmse,
    'test-rmse': test_rmse,
})