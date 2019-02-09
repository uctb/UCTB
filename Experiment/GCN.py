import os
import datetime
import numpy as np

from local_path import data_dir, tf_model_dir
from DataPreprocess.UtilClass import MoveSample, SplitData
from Model.GraphModelLayers import GraphBuilder
from Model.GCN_Regression import GCNRegression
from Train.EarlyStopping import EarlyStoppingTTest
from Train.MiniBatchTrain import MiniBatchTrainMultiData
from EvalClass.Accuracy import Accuracy
from DataSet.utils import is_work_day
from dateutil.parser import parse


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--City', default='NYC')
    # network parameter
    parser.add_argument('--K', default='1')
    parser.add_argument('--L', default='1')
    # training parameters
    parser.add_argument('--Epoch', default='5000')
    parser.add_argument('--Train', default='True')
    parser.add_argument('--lr', default='1e-4')
    parser.add_argument('--patience', default='20')
    parser.add_argument('--BatchSize', default='128')
    # device parameter
    parser.add_argument('--Device', default='0')
    # version contral
    parser.add_argument('--CodeVersion', default='0')
    return parser


parser = parameter_parser()
args = parser.parse_args()

city =  args.City

traffic_data_file = '%s_Graph.npy' % city
weather_file = '%s_Weather.npy' % city

time_range = ['2013-07-01', '2017-09-30']

# traffic feature
traffic_data = np.load(os.path.join(data_dir, traffic_data_file))
traffic_data = traffic_data[:, np.where(np.mean(traffic_data, axis=0)*24 > 1)[0]]
# external feature
weather_data = np.load(os.path.join(data_dir, weather_file))
day_list = [[1 if is_work_day(parse(time_range[1]) + datetime.timedelta(hours=e)) else 0]\
            for e in range((parse(time_range[1]) - parse(time_range[0])).days * 24)]
external_feature = np.concatenate((weather_data, day_list), axis=-1)

station_number = traffic_data.shape[1]
external_dim = external_feature.shape[1]

train_data, val_data, test_data = SplitData.split_data(traffic_data, 0.8, 0.1, 0.1)
train_ef, val_ef, test_ef =  SplitData.split_data(external_feature, 0.8, 0.1, 0.1)

closeness_move_sample = MoveSample(feature_step=1, feature_stride=1, feature_length=6, target_length=1)

train_x, train_y = closeness_move_sample.general_move_sample(train_data)
train_ef = train_ef[-len(train_x)-1:-1]

val_x, val_y = closeness_move_sample.general_move_sample(val_data)
val_ef = val_ef[-len(val_x)-1:-1]

test_x, test_y = closeness_move_sample.general_move_sample(test_data)
test_ef = test_ef[-len(test_x)-1:-1]

# reshape
train_x = train_x.transpose([0, 2, 3, 1])
val_x = val_x.transpose([0, 2, 3, 1])
test_x = test_x.transpose([0, 2, 3, 1])

train_y = train_y.reshape([-1, station_number])
val_y = val_y.reshape([-1, station_number])
test_y = test_y.reshape([-1, station_number])

LM = GraphBuilder.correlation_graph(train_data[-30*24:], keep_weight=False)

# parse parameters
train = True if args.Train == 'True' else False
K = int(args.K)
L = int(args.L)
lr = float(args.lr)
patience = int(args.patience)
num_epoch = int(args.Epoch)
batch_size = int(args.BatchSize)

GPU_DEVICE = args.Device

code_version = 'GCN_{}_K{}L{}_{}'.format(city, K, L, args.CodeVersion)

GCNRegression_Obj = GCNRegression(num_node=station_number, GCN_K=K, GCN_layers=L, external_dim=external_dim,
                                  T=6, num_filter_conv1x1=32, num_hidden_units=64,
                                  lr=lr, code_version=code_version, GPU_DEVICE=GPU_DEVICE, model_dir=tf_model_dir)

GCNRegression_Obj.build()

de_normalizer = None

if train:

    val_rmse_record = []
    test_rmse_record = []

    early_stop = EarlyStoppingTTest(length=patience, p_value_threshold=0.1)

    # best_record = None

    for epoch in range(num_epoch):

        mini_batch_train = MiniBatchTrainMultiData([train_x, train_y, train_ef], batch_size=batch_size)

        loss_list = []

        for i in range(mini_batch_train.num_batch):
            X, y, ef = mini_batch_train.get_batch()

            l = GCNRegression_Obj.fit(X, y.reshape([-1, station_number]), LM, external_feature=ef)

            loss_list.append(l)

        # validation
        val_rmse, = GCNRegression_Obj.evaluate(val_x, val_y, LM, external_feature=val_ef,
                                               metric=[Accuracy.RMSE], threshold=0, de_normalizer=de_normalizer)

        # test
        test_rmse, = GCNRegression_Obj.evaluate(test_x, test_y, LM, external_feature=test_ef,
                                                metric=[Accuracy.RMSE], threshold=0, de_normalizer=de_normalizer)

        val_rmse_record.append([float(val_rmse)])
        test_rmse_record.append([float(test_rmse)])

        if early_stop.stop(val_rmse):
            break

        avg_loss = '%.5f' % np.mean(loss_list)
        val_rmse = '%.5f' % val_rmse
        test_rmse = '%.5f' % test_rmse

        print(code_version, epoch, avg_loss, val_rmse, test_rmse)

    GCNRegression_Obj.save(code_version)

GCNRegression_Obj.load(code_version)

# test
test_rmse, = GCNRegression_Obj.evaluate(test_x, test_y, LM, external_feature=test_ef,
                                        metric=[Accuracy.RMSE], threshold=0, de_normalizer=de_normalizer)

print('########################################################################')
print(code_version, 'Test RMSE', test_rmse)
print('########################################################################')