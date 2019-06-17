import nni
import numpy as np

from UCTB.dataset import NodeTrafficLoader_CPT
from sklearn.ensemble import GradientBoostingRegressor
from UCTB.evaluation import metric

dataset = 'ChargeStation'
city = 'Beijing'

params = nni.get_next_parameter()

data_loader = NodeTrafficLoader_CPT(dataset=dataset, city=city, with_lm=False,
                                    C_T=params['CT'], P_T=params['PT'], T_T=params['TT'],
                                    test_ratio=0.1, normalize=True)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = GradientBoostingRegressor(n_estimators=params['num_estimator'], max_depth=params['max_depth'])

    train_x = np.concatenate([data_loader.train_closeness[:, 0, i, :],
                              data_loader.train_period[:, 0, i, :],
                              data_loader.train_trend[:, 0, i, :]], axis=-1)

    test_x = np.concatenate([data_loader.test_closeness[:, 0, i, :],
                             data_loader.test_period[:, 0, i, :],
                             data_loader.test_trend[:, 0, i, :]], axis=-1)

    model.fit(train_x, data_loader.train_y[:, i])

    p = model.predict(test_x).reshape([-1, 1, 1])

    prediction.append(p)

prediction = np.concatenate(prediction, axis=-2)

print(dataset, city, 'RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))
print(dataset, city, 'MAPE', metric.mape(prediction, data_loader.test_y, threshold=0))


def show_prediction(prediction, target, station_index, start=0, end=-1):

    import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    # axs[0].plot(prediction[start:end, station_index])
    # axs[1].plot(target[start:end, station_index])

    plt.plot(prediction[start:end, station_index], 'b')
    plt.plot(target[start:end, station_index], 'r')

    print(metric.rmse(prediction[start:end, station_index], target[start:end, station_index]))

    print(prediction[start:end, station_index].max(), target[start:end, station_index].max())
    print(prediction[start:end, station_index].min(), target[start:end, station_index].min())

    plt.show()