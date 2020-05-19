import numpy as np
import argparse
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric
from UCTB.preprocess import SplitData
import nni
import os

params = {
    'CT': 12,
    'PT': 6,
    'TT': 4,
    'max_depth': 8,
    'num_boost_round': 51
}

parser = argparse.ArgumentParser(description="Argument Parser")
# data source
parser.add_argument('--dataset', default='Metro', type=str)
parser.add_argument('--city', default="Chongqing", type=str)
parser.add_argument('--MergeIndex', default=3)
parser.add_argument('--DataRange', default="all")
parser.add_argument('--TrainDays', default="all")

#use params and args to show its difference
args = vars(parser.parse_args())


params.update(nni.get_next_parameter())

data_loader = NodeTrafficLoader(dataset=args["dataset"], city=args['city'], closeness_len=int(params['CT']), period_len=int(params['PT']), trend_len=int(params['TT']),
                                data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                test_ratio=0.1,
                                with_lm=False, normalize=False, MergeIndex=args['MergeIndex'],
                                MergeWay="max" if args["dataset"] == "ChargeStation" else "sum")

train_closeness, val_closeness = SplitData.split_data(
    data_loader.train_closeness, [0.9, 0.1])
train_period, val_period = SplitData.split_data(
    data_loader.train_period, [0.9, 0.1])
train_trend, val_trend = SplitData.split_data(
    data_loader.train_trend, [0.9, 0.1])

train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

prediction_test = []
prediction_val = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(n_estimators=int(
        params['num_boost_round']), max_depth=int(params['max_depth']))

    X_Train = []
    X_Val = []
    X_Test = []
    if int(params['CT']) > 0:
        X_Train.append(train_closeness[:, i, :, 0])
        X_Val.append(val_closeness[:, i, :, 0])
        X_Test.append(data_loader.test_closeness[:, i, :, 0])
    if int(params['PT']) > 0:
        X_Train.append(train_period[:, i, :, 0])
        X_Val.append(val_period[:, i, :, 0])
        X_Test.append(data_loader.test_period[:, i, :, 0])
    if int(params['TT']) > 0:
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



nni.report_final_result({'default': metric.rmse(prediction_val, val_y, threshold=0),
                         'test-rmse': metric.rmse(prediction_test, data_loader.test_y, threshold=0)})
