from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
import argparse
from UCTB.evaluation import metric
from UCTB.preprocess import SplitData
import nni
import os


params = {
    'CT': 0,
    'PT': 0,
    'TT': 4,
}

params.update(nni.get_next_parameter())


# acquire data source path
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('--dataset', default='Metro', type=str)
parser.add_argument('--city', default="Shanghai", type=str)
parser.add_argument('--MergeIndex', default=1)
parser.add_argument('--DataRange', default="all")
parser.add_argument('--TrainDays', default="all")
parser.add_argument('--MergeWay', default="sum")
parser.add_argument('--test_ratio', default=0.1, type=float)

# note that the args is different from param
args = vars(parser.parse_args())


data_loader = NodeTrafficLoader(dataset=args["dataset"], city=args['city'], closeness_len=int(params['CT']), period_len=int(params['PT']), trend_len=int(params['TT']),
                                data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                test_ratio=args['test_ratio'],
                                with_lm=False, normalize=False, MergeIndex=args['MergeIndex'],
                                MergeWay=args['MergeWay'])

train_closeness, val_closeness = SplitData.split_data(
    data_loader.train_closeness, [0.9, 0.1])
train_period, val_period = SplitData.split_data(
    data_loader.train_period, [0.9, 0.1])
train_trend, val_trend = SplitData.split_data(
    data_loader.train_trend, [0.9, 0.1])


train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])


hm_obj = HM(c=data_loader.closeness_len,
            p=data_loader.period_len, t=data_loader.trend_len)


test_prediction = hm_obj.predict(closeness_feature=data_loader.test_closeness,
                                 period_feature=data_loader.test_period,
                                 trend_feature=data_loader.test_trend)

val_prediction = hm_obj.predict(closeness_feature=val_closeness,
                                period_feature=val_period,
                                trend_feature=val_trend)


print('Test RMSE', metric.rmse(test_prediction, data_loader.test_y, threshold=0))
print('Val RMSE', metric.rmse(val_prediction, val_y, threshold=0))


nni.report_final_result({'default': metric.rmse(val_prediction, val_y, threshold=0),
                         'test-rmse': metric.rmse(test_prediction, data_loader.test_y, threshold=0)})
