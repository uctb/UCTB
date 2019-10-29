import nni
import numpy as np
import argparse

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric


parser = argparse.ArgumentParser(description="Argument Parser")
# data source
parser.add_argument('--Dataset', default='Metro')
parser.add_argument('--City', default='Shanghai')
# network parameter
parser.add_argument('--CT', default='168', type=int)

parser.add_argument('--ar', default='6', type=int)
parser.add_argument('--d', default='0', type=int)
parser.add_argument('--ma', default='1', type=int)

parser.add_argument('--sar', default='0', type=int)
parser.add_argument('--sd', default='0', type=int)
parser.add_argument('--sma', default='0', type=int)
parser.add_argument('--sp', default='0', type=int)

parser.add_argument('--DataRange', default='All')
parser.add_argument('--TrainDays', default='60')

args = vars(parser.parse_args())

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)

data_loader = NodeTrafficLoader(dataset=args['Dataset'], city=args['City'],
                                closeness_len=int(args['CT']), period_len=0, trend_len=0,
                                data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                with_lm=False, with_tpe=False, normalize=False)

test_rmse_collector = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print(args['Dataset'], args['City'], 'Station', i, 'total', data_loader.station_number)

    try:
        model_obj = ARIMA(time_sequence=data_loader.train_closeness[:, i, -1, 0],
                          order=[args['ar'], args['d'], args['ma']],
                          seasonal_order=[args['sar'], args['sd'], args['sma'], args['sp']])

        test_prediction = model_obj.predict(time_sequences=data_loader.test_closeness[:, i, :, 0],
                                            forecast_step=1)

    except Exception as e:
        print('Converge failed with error', e)
        print('Using last as prediction')

        test_prediction = data_loader.test_closeness[:, i, -1:, :]

    test_rmse_collector.append(test_prediction)

    print('Station', i, metric.rmse(test_prediction, data_loader.test_y[:, i:i+1], threshold=0))

test_rmse_collector = np.concatenate(test_rmse_collector, axis=-2)
test_rmse = metric.rmse(test_rmse_collector, data_loader.test_y, threshold=0)

print(args['Dataset'], args['City'], 'test_rmse', test_rmse)