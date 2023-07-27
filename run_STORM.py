import sys, os

from units.data_loader import NodeTrafficLoader
from units.GraphGenerator import GraphGenerator
from UCTB.evaluation import metric
from model.STORM import STORM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE
import argparse
import time

# python DirRec_MT_origin.py --dataset hospital --datatype hour --gpu 0

# args
parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='violation')
parser.add_argument('--datatype', type=str, default='XM')
parser.add_argument('--n_pred', type=str, default=12)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--gcn_k', type=int, default=1)  # highest order of Chebyshev Polynomial approximation in GCN
parser.add_argument('--gcn_l', type=int, default=2)  # numbers of GCN layer
parser.add_argument('--gclstm_l', type=int, default=1)  # numbers of GCN-LSTM layer
parser.add_argument('--gru_l', type=int, default=1)  # numbers of GRU layer
args = parser.parse_args()

# params
n_pred = int(args.n_pred)
gcn_k = int(args.gcn_k)
gcn_layers = int(args.gcn_l)
gclstm_layers = int(args.gclstm_l)
gru_layers = int(args.gru_l)
model_name = 'STORM'
gpu_device = args.gpu
model_dir = 'outputs/'
code_version = model_name + '_' + args.dataset + '_' + args.datatype + '_' + str(args.gclstm_l)

# path
data_path = 'data/'
output_path = 'outputs/multi_step/'
pkl_file_name = data_path + args.dataset + '_' + args.datatype + '.pkl'
log_name = output_path + model_name + '/' + args.dataset + '_' + args.datatype + '.log'

# data_loader
data_loader = NodeTrafficLoader(dataset=pkl_file_name, closeness_len=24, target_length=n_pred,
                                test_ratio=0.2)

# Build Graph
graph_obj = GraphGenerator(graph='Correlation-Function-Distance', data_loader=data_loader)

start = time.time()

# 定义H个模型分别进行训练
model_list = []
temp_traffic_predict = []
temp_external_predict = []
temp_closeness_feature = data_loader.train_closeness
# temp_external_feature shape is [time_slots, external_dim, n_pred, 1]
temp_external_feature = data_loader.train_tef_y
temp_external_feature = np.repeat(temp_external_feature, data_loader.station_number, axis=1)
temp_external_feature = np.expand_dims(temp_external_feature, axis=-1)
spatial_external_feature = data_loader.train_sef_y
# spatial_external_feature = np.repeat(spatial_external_feature, data_loader.station_number, axis=1)
spatial_external_feature = np.expand_dims(spatial_external_feature, axis=-1)

for i in range(n_pred):
    code_version = model_name + '_' + args.dataset + '_' + args.datatype + '_' + str(args.gclstm_l) + '_step' + str(i)
    temp_closeness_len = data_loader.closeness_len + i
    # 输入的热点数据的维度大小
    temp_external_len = n_pred + i
    if i != 0:
        temp_closeness_feature = np.concatenate((temp_closeness_feature, temp_traffic_predict), axis=2)
        temp_external_feature = np.concatenate((temp_external_feature, temp_external_predict), axis=2)
        # spatial_external_feature = np.concatenate((spatial_external_feature, temp_external_predict), axis=2)
    temp_model = STORM(num_node=data_loader.station_number,
                       station_num=data_loader.station_number,
                       external_dim=data_loader.external_dim,
                       external_len=temp_external_len,
                       closeness_len=temp_closeness_len,
                       period_len=data_loader.period_len,
                       trend_len=0,
                       num_graph=graph_obj.LM.shape[0],
                       gcn_k=gcn_k,
                       gcn_layers=gcn_layers,
                       gclstm_layers=gclstm_layers,
                       gru_layers=gcn_layers,
                       num_hidden_units=64,
                       num_dense_units=32,
                       lr=2e-3,
                       loss_w_node=0.8,
                       loss_w_tef=0.2,
                       code_version=code_version,
                       model_dir=model_dir,
                       gpu_device=gpu_device)
    # tf-graph
    temp_model.build()
    # training
    temp_model.fit(laplace_matrix=graph_obj.LM,
                   closeness_traffic_feature=temp_closeness_feature,
                   period_traffic_feature=data_loader.train_period,
                   target_traffic=data_loader.train_y[:, :, i].reshape((-1, data_loader.station_number, 1)),
                   temporal_external_feature=temp_external_feature,
                   event_impulse_response=data_loader.train_eir_y[:, :, i].reshape((-1, data_loader.station_number, 1)).repeat(data_loader.external_dim, 1),
                   spatial_external_feature=spatial_external_feature,
                   early_stop_length=50,

                   sequence_length=data_loader.train_sequence_len,
                   output_names=['loss', 'prediction', 'external_prediction'],
                   max_epoch=500,
                   auto_load_model=True)
    # save
    model_list.append(temp_model)
    # prediction
    temp_predict = temp_model.predict(
                    laplace_matrix=graph_obj.LM,
                    closeness_traffic_feature=temp_closeness_feature,
                    period_traffic_feature=data_loader.train_period,
                    target_traffic=data_loader.train_y[:, :, i].reshape((-1, data_loader.station_number, 1)),
                    temporal_external_feature=temp_external_feature,
                    event_impulse_response=data_loader.train_eir_y[:, :, i].reshape((-1, data_loader.station_number, 1)).repeat(data_loader.external_dim, 1),
                    spatial_external_feature=spatial_external_feature,
                    output_names=['prediction', 'external_prediction'],
                    sequence_length=data_loader.train_sequence_len
    )

    temp_traffic_predict = temp_predict['prediction']
    temp_traffic_predict = temp_traffic_predict.reshape(
        (temp_traffic_predict.shape[0], temp_traffic_predict.shape[1], 1, 1))
    temp_external_predict = temp_predict['external_prediction']
    temp_external_predict = temp_external_predict.reshape(
        (temp_external_predict.shape[0], temp_external_predict.shape[1], 1, 1))

# 使用H个模型预测得到未来H步
predict_list = []
external_predict_list = []
temp_traffic_predict = []
temp_external_predict = []
temp_closeness_feature = data_loader.test_closeness
temp_external_feature = data_loader.test_tef_y.repeat(data_loader.station_number, 1)
temp_external_feature = np.expand_dims(temp_external_feature, axis=-1)
spatial_external_feature = data_loader.test_sef_y
# spatial_external_feature = np.repeat(spatial_external_feature, data_loader.station_number, axis=1)
spatial_external_feature = np.expand_dims(spatial_external_feature, axis=-1)
for i in range(n_pred):
    temp_model = model_list[i]
    if i != 0:
        temp_closeness_feature = np.concatenate((temp_closeness_feature, temp_traffic_predict), axis=2)
        temp_external_feature = np.concatenate((temp_external_feature, temp_external_predict), axis=2)
    temp_predict = temp_model.predict(
                laplace_matrix=graph_obj.LM,
                closeness_traffic_feature=temp_closeness_feature,
                period_traffic_feature=data_loader.test_period,
                target_traffic=data_loader.test_y[:, :, i].reshape((-1, data_loader.station_number, 1)),
                temporal_external_feature=temp_external_feature,
                event_impulse_response=data_loader.test_eir_y[:, :, i].reshape((-1, data_loader.station_number, 1)).repeat(data_loader.external_dim, 1),
                spatial_external_feature=spatial_external_feature,
                output_names=['prediction', 'external_prediction'],
                sequence_length=data_loader.test_sequence_len
    )
    temp_traffic_predict = temp_predict['prediction']
    if i == 0:
        predict_list = temp_traffic_predict
    else:
        predict_list = np.concatenate((predict_list, temp_traffic_predict), axis=2)
    temp_traffic_predict = temp_traffic_predict.reshape(
        (temp_traffic_predict.shape[0], temp_traffic_predict.shape[1], 1, 1))

    temp_external_predict = temp_predict['external_prediction']
    if i == 0:
        external_predict_list = temp_external_predict
    else:
        external_predict_list = np.concatenate((external_predict_list, temp_external_predict), axis=2)
    temp_external_predict = temp_external_predict.reshape(
        (temp_external_predict.shape[0], temp_external_predict.shape[1], 1, 1))

external_prediction = external_predict_list
external_target = data_loader.test_tef_y.repeat(data_loader.station_number, 1)


# 对每一步得到一个评估结果
# Evaluation
print('Total time cost is %.3f' % float(time.time() - start))
# eval
prediction = data_loader.normalizer.min_max_denormal(predict_list)
prediction = np.where(prediction > 0, prediction, 0)
target = data_loader.normalizer.min_max_denormal(data_loader.test_y)

evaluation_result = pd.DataFrame(columns=["MAE", "RMSE", "MAPE"], index=range(1, n_pred + 1))
for i in range(n_pred):
    # reshape
    cur_prediction = prediction[:, :, i]
    cur_target = target[:, :, i]
    # result
    mae = MAE(cur_prediction, cur_target)
    rmse = metric.rmse(cur_prediction, cur_target, threshold=0)
    mape = metric.mape(cur_prediction, cur_target, threshold=0.1)
    # save
    evaluation_result.loc[i + 1, "MAE"] = mae
    evaluation_result.loc[i + 1, "RMSE"] = rmse
    evaluation_result.loc[i + 1, "MAPE"] = mape
    # print
    print("Step %02d, MAE: %.4f, RMSE: %.4f, MAPE:%.4f" % (i + 1, mae, rmse, mape))

# plot
for eva in evaluation_result.columns:
    fig = plt.figure(0, dpi=300, figsize=(8, 5))
    plt.plot(evaluation_result.index, evaluation_result.loc[:, eva])
    plt.xlabel("Horizon")
    plt.ylabel(eva)
    plt.savefig(
        output_path + model_name + '/' + args.dataset + '_' + args.datatype + '/figure/' + 'multi-step_' + eva + '.png')
    plt.close(0)
# save
np.save(output_path + model_name + '/' + args.dataset + '_' + args.datatype + '/prediction.npy', prediction)
np.save(output_path + model_name + '/' + args.dataset + '_' + args.datatype + '/target.npy', target)
evaluation_result.to_csv(output_path + model_name + '/' + args.dataset + '_' + args.datatype + '/evaluation.csv', float_format="%.4f")

