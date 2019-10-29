import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

closeness_len = 6
period_len = 7
trend_len = 4

data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian',
                                closeness_len=closeness_len, period_len=period_len, trend_len=trend_len,
                                with_lm=False, normalize=False)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3)

    X_Train = []
    X_Test = []
    if closeness_len > 0:
        X_Train.append(data_loader.train_closeness[:, i, :, 0])
        X_Test.append(data_loader.test_closeness[:, i, :, 0])
    if period_len > 0:
        X_Train.append(data_loader.train_period[:, i, :, 0])
        X_Test.append(data_loader.test_period[:, i, :, 0])
    if trend_len > 0:
        X_Train.append(data_loader.train_trend[:, i, :, 0])
        X_Test.append(data_loader.test_trend[:, i, :, 0])

    X_Train = np.concatenate(X_Train, axis=-1)
    X_Test = np.concatenate(X_Test, axis=-1)

    model.fit(X_Train, data_loader.train_y[:, i, 0])

    p = model.predict(X_Test)

    prediction.append(p.reshape([-1, 1, 1]))

prediction = np.concatenate(prediction, axis=-2)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))