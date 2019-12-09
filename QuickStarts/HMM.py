import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HMM
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='Chicago',
                                closeness_len=12, period_len=0, trend_len=0,
                                with_lm=False, normalize=False)

prediction = []
for station_index in range(data_loader.station_number):
    # train the hmm model
    try:
        hmm = HMM(num_components=8, n_iter=100)
        hmm.fit(data_loader.train_closeness[:, station_index:station_index+1, -1, 0])
        # predict
        p = []
        for time_index in range(data_loader.test_closeness.shape[0]):
            p.append(hmm.predict(data_loader.test_closeness[time_index, station_index, :, :], length=1))
    except Exception as e:
        print('Failed at station', station_index, 'with error', e)
        # using zero as prediction
        p = [[[0]] for _ in range(data_loader.test_closeness.shape[0])]

    prediction.append(np.array(p)[:, :, 0])
    print('Node', station_index, 'finished')

prediction = np.array(prediction).transpose([1, 0, 2])
print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))