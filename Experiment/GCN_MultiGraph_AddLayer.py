import os
import datetime
import numpy as np

from local_path import data_dir, tf_model_dir
from DataPreprocess.UtilClass import MoveSample, SplitData
from Model.GraphModelLayers import GraphBuilder
from Model.MGCN_Regression_AddLayer import MGCNRegression
from Train.EarlyStopping import EarlyStoppingTTest
from Train.MiniBatchTrain import MiniBatchTrainMultiData
from EvalClass.Accuracy import Accuracy
from DataSet.utils import is_work_day
from dateutil.parser import parse

from Experiment.data_loader import gcn_data_loader


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--City', default='DC')
    # network parameter
    parser.add_argument('--K', default='1')
    parser.add_argument('--L', default='1')
    parser.add_argument('--Graph', default='Distance')
    parser.add_argument('--GLL', default='2')

    parser.add_argument('--TrainDays', default='All')

    # Graph parameter
    parser.add_argument('--TC', default='0')
    parser.add_argument('--TD', default='1000')
    parser.add_argument('--TI', default='500')

    # training parameters
    parser.add_argument('--Epoch', default='5000')
    parser.add_argument('--Train', default='True')
    parser.add_argument('--lr', default='1e-4')
    parser.add_argument('--patience', default='20')
    parser.add_argument('--BatchSize', default='64')
    # device parameter
    parser.add_argument('--Device', default='1')
    # version contral
    parser.add_argument('--CodeVersion', default='Debug')
    return parser


parser = parameter_parser()
args = parser.parse_args()

data_loader = gcn_data_loader(args)

print(data_loader.train_data.shape)

# parse parameters
train = True if args.Train == 'True' else False
K = [int(e) for e in args.K.split(',') if len(e) > 0]
L = [int(e) for e in args.L.split(',') if len(e) > 0]
lr = float(args.lr)
patience = int(args.patience)
num_epoch = int(args.Epoch)
batch_size = int(args.BatchSize)

GPU_DEVICE = args.Device

code_version = 'MGCN_{}_{}_GL{}K{}L{}_{}'.format(data_loader.city, ''.join([e[0] for e in args.Graph.split('-')]),
                                                 int(args.GLL),
                                                 ''.join([str(e) for e in K]),
                                                 ''.join([str(e) for e in L]), args.CodeVersion)

MGCNRegression_Obj = MGCNRegression(num_node=data_loader.station_number, GCN_K=K, GCN_layers=L,
                                    GLL=int(args.GLL),
                                    num_graph=data_loader.LM.shape[0],
                                    external_dim=data_loader.external_dim,
                                    T=6, num_filter_conv1x1=32, num_hidden_units=64,
                                    lr=lr, code_version=code_version, GPU_DEVICE=GPU_DEVICE, model_dir=tf_model_dir)

MGCNRegression_Obj.build()

de_normalizer = None

if train:

    try:
        MGCNRegression_Obj.load(code_version)
    except Exception as e:
        print(e)

    val_rmse_record = []
    test_rmse_record = []

    early_stop = EarlyStoppingTTest(length=patience, p_value_threshold=0.1)

    best_record = None

    for epoch in range(num_epoch):

        mini_batch_train = MiniBatchTrainMultiData([data_loader.train_x, data_loader.train_y, data_loader.train_ef],
                                                   batch_size=batch_size)

        loss_list = []

        for i in range(mini_batch_train.num_batch):
            X, y, ef = mini_batch_train.get_batch()

            l = MGCNRegression_Obj.fit(X, y.reshape([-1, data_loader.station_number]), data_loader.LM, external_feature=ef)

            loss_list.append(l)

        # validation
        val_rmse, = MGCNRegression_Obj.evaluate(data_loader.val_x, data_loader.val_y, data_loader.LM,
                                                external_feature=data_loader.val_ef,
                                                metric=[Accuracy.RMSE], threshold=0, de_normalizer=de_normalizer)

        # test
        test_rmse, = MGCNRegression_Obj.evaluate(data_loader.test_x, data_loader.test_y, data_loader.LM,
                                                 external_feature=data_loader.test_ef,
                                                 metric=[Accuracy.RMSE], threshold=0, de_normalizer=de_normalizer)

        val_rmse_record.append([float(val_rmse)])
        test_rmse_record.append([float(test_rmse)])

        if early_stop.stop(val_rmse):
            break

        if best_record is None or val_rmse < best_record:
            best_record = val_rmse
            MGCNRegression_Obj.save(code_version)

        avg_loss = '%.5f' % np.mean(loss_list)
        val_rmse = '%.5f' % val_rmse
        test_rmse = '%.5f' % test_rmse

        print(code_version, epoch, avg_loss, val_rmse, test_rmse)

MGCNRegression_Obj.load(code_version)

# test
test_rmse, = MGCNRegression_Obj.evaluate(data_loader.test_x, data_loader.test_y, data_loader.LM,
                                        external_feature=data_loader.test_ef,
                                        metric=[Accuracy.RMSE], threshold=0, de_normalizer=de_normalizer)

print('########################################################################')
print(code_version, 'Test RMSE', test_rmse)
print('########################################################################')