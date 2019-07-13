from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HMM
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='ChargeStation', city='Beijing',
                                closeness_len=6, period_len=7, trend_len=4,
                                with_lm=False, normalize=False)

prediction = []

for station_index in range(data_loader.station_number):

    # train the hmm model
    try:
        hmm = HMM(num_components=16, n_iter=1000)
        hmm.fit(data_loader.train_closeness[:, station_index:station_index+1, -1, 0])
        # predict
        p = []
        for time_index in range(data_loader.test_closeness.shape[0]):
            p.append(hmm.predict(data_loader.test_closeness[time_index, station_index, :, :], length=1))
    except Exception as e:
        print('Failed at station', station_index, 'with error', e)
        p = [[[0]] for _ in range(data_loader.test_closeness.shape[0])]

    prediction.append(p)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))