import numpy as np
import argparse
from tqdm import tqdm
from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric
from UCTB.preprocess import SplitData
import os

import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Argument Parser")
# data source
parser.add_argument('--dataset', default=' ', type=str)
parser.add_argument('--city', default=None)
parser.add_argument('--MergeIndex', default=3)

# network parameter
parser.add_argument('--CT', default='168', type=int)

parser.add_argument('--ar', default='3', type=int)
parser.add_argument('--d', default='0', type=int)
parser.add_argument('--ma', default='1', type=int)

parser.add_argument('--sar', default='0', type=int)
parser.add_argument('--sd', default='0', type=int)
parser.add_argument('--sma', default='0', type=int)
parser.add_argument('--sp', default='0', type=int)


parser.add_argument('--DataRange', default="all")
parser.add_argument('--TrainDays', default="all")


args = vars(parser.parse_args())


data_loader = NodeTrafficLoader(dataset=args['dataset'], city=args['city'],
                                data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                test_ratio=0.1,
                                closeness_len=int(args['CT']), period_len=0, trend_len=0,
                                with_lm=False, with_tpe=False, normalize=False,MergeIndex=args['MergeIndex'],
                                MergeWay="max" if args["dataset"] == "ChargeStation" else "sum")
                                

train_closeness, val_closeness = SplitData.split_data(data_loader.train_closeness, [0.9, 0.1])
train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

val_prediction_collector = []
test_prediction_collector = []


print('*************************************************************')

for i in tqdm(range(data_loader.station_number)):

    try:
        model_obj = ARIMA(time_sequence=train_closeness[:, i, -1, 0],
                          order=[args['ar'], args['d'], args['ma']],
                          seasonal_order=[args['sar'], args['sd'], args['sma'], args['sp']])

        val_prediction = model_obj.predict(
            time_sequences=val_closeness[:, i, :, 0], forecast_step=1)
        test_prediction = model_obj.predict(
            time_sequences=data_loader.test_closeness[:, i, :, 0], forecast_step=1)

    except Exception as e:
        print('Converge failed with error', e)
        print('Using last as prediction')

        val_prediction = val_closeness[:, i, -1:, :]
        test_prediction = data_loader.test_closeness[:, i, -1:, :]

    val_prediction_collector.append(val_prediction)
    test_prediction_collector.append(test_prediction)

    print('Station', i, metric.rmse(test_prediction,
                                    data_loader.test_y[:, i:i+1], threshold=0))


val_prediction_collector = np.concatenate(val_prediction_collector, axis=-2)
test_prediction_collector = np.concatenate(test_prediction_collector, axis=-2)

val_rmse = metric.rmse(val_prediction_collector, val_y, threshold=0)
test_rmse = metric.rmse(test_prediction_collector,
                        data_loader.test_y, threshold=0)

print(args['dataset'], args['city'], 'val_rmse', val_rmse)
print(args['dataset'], args['city'],'test_rmse', test_rmse)


print('*************************************************************')
