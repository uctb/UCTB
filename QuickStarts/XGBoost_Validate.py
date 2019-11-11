import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric
from UCTB.preprocess import SplitData

closeness_len = 6
period_len = 7
trend_len = 4

data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian',
                                closeness_len=closeness_len, period_len=period_len, trend_len=trend_len,
                                with_lm=False, normalize=False)

train_closeness, val_closeness = SplitData.split_data(data_loader.train_closeness, [0.9, 0.1])
train_period, val_period = SplitData.split_data(data_loader.train_period, [0.9, 0.1])
train_trend, val_trend = SplitData.split_data(data_loader.train_trend, [0.9, 0.1])

train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

prediction_test = []
prediction_val = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(n_estimators=100, max_depth=3, objective='reg:linear')

    X_Train = []
    X_Val = []
    X_Test = []
    if closeness_len > 0:
        X_Train.append(train_closeness[:, i, :, 0])
        X_Val.append(val_closeness[:, i, :, 0])
        X_Test.append(data_loader.test_closeness[:, i, :, 0])
    if period_len > 0:
        X_Train.append(train_period[:, i, :, 0])
        X_Val.append(val_period[:, i, :, 0])
        X_Test.append(data_loader.test_period[:, i, :, 0])
    if trend_len > 0:
        X_Train.append(train_trend[:, i, :, 0])
        X_Val.append(val_trend[:, i, :, 0])
        X_Test.append(data_loader.test_trend[:, i, :, 0])

    X_Train = np.concatenate(X_Train, axis=-1)
    X_Val = np.concatenate(X_Val, axis=-1)
    X_Test = np.concatenate(X_Test, axis=-1)

    model.fit(X_Train, train_y[:, i, 0])

    p_val = model.predict(X_Val)
    p_test = model.predict(X_Test)

    prediction_test.append(p_test.reshape([-1, 1, 1]))
    prediction_val.append(p_val.reshape([-1, 1, 1]))

prediction_test = np.concatenate(prediction_test, axis=-2)
prediction_val = np.concatenate(prediction_val, axis=-2)

print('Val RMSE', metric.rmse(prediction_val, val_y, threshold=0))
print('Test RMSE', metric.rmse(prediction_test, data_loader.test_y, threshold=0))