import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric


# params
dataset_name = "Bike_NYC"
model_name = "DirRec_XGBoost"
output_path = "../Outputs/"+model_name+"-"+dataset_name
model_dir = "../Outputs/model_dir"
code_version = model_name+"-"+dataset_name
batch_size = 64
n_pred = 12
gpu_device = '0'

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=6, period_len=7, trend_len=4, target_length=n_pred, with_lm=False, normalize=False)

start = time.time()

# define (station_number * n_pred) model to train
model_list = []
temp_predict = []
node_predict = []
temp_trainX = np.concatenate((data_loader.train_trend, data_loader.train_period, data_loader.train_closeness), axis = 2)
for i in range(n_pred):
    model_list.append([])
    if i != 0:
        temp_trainX = np.concatenate((temp_trainX,temp_predict), axis=2)
    temp_predict = []
    for j in range(data_loader.station_number):
        print('Step %d, Station %d' % (i, j))
        # define
        temp_model = XGBoost(n_estimators=100, max_depth=3, objective='reg:squarederror')
        # train
        temp_model.fit(temp_trainX[:, j, :, 0], data_loader.train_y[:, j, i])
        # save
        model_list[i].append(temp_model)
        # predict
        node_predict = temp_model.predict(temp_trainX[:, j, :, 0])
        temp_predict.append(node_predict.reshape((-1, 1, 1)))
    # temp_predict shape is [sequence_len, station_number, 1]
    temp_predict = np.concatenate(temp_predict, axis=1)
    temp_predict = temp_predict.reshape((temp_predict.shape[0], temp_predict.shape[1], 1, 1))


# 使用H个模型预测得到未来H步
predict_list = []
temp_predict = []
node_predict = []
temp_testX = np.concatenate((data_loader.test_trend, data_loader.test_period, data_loader.test_closeness), axis = 2)
for i in range(n_pred):
    if i != 0:
        temp_testX = np.concatenate((temp_testX,temp_predict), axis=2)
    temp_predict = []
    for j in range(data_loader.station_number):
        temp_model = model_list[i][j]
        node_predict = temp_model.predict(temp_testX[:, j, :, 0])
        temp_predict.append(node_predict.reshape((-1, 1, 1)))
    # temp_predict shape is [sequence_len, station_number, 1]
    temp_predict = np.concatenate(temp_predict, axis=1)
    predict_list.append(temp_predict)
    temp_predict = temp_predict.reshape((temp_predict.shape[0], temp_predict.shape[1], 1, 1))

print('Total time cost is %.3f' % float(time.time()-start))
# Evaluation
predict_list = np.concatenate(predict_list, axis=2)
prediction = np.where(predict_list>0, predict_list, 0)
target = data_loader.test_y
evaluation_result = pd.DataFrame(columns=["MAE", "RMSE", "MAPE"], index=range(1, n_pred+1))
for i in range(n_pred):
    # reshape
    cur_prediction = prediction[:,:,i]
    cur_target = target[:,:,i]
    # result
    mae = MAE(cur_prediction, cur_target)
    rmse = metric.rmse(cur_prediction, cur_target)
    mape = metric.mape(cur_prediction, cur_target, threshold=0.1)
    # save
    evaluation_result.loc[i+1, "MAE"] = mae
    evaluation_result.loc[i+1, "RMSE"] = rmse
    evaluation_result.loc[i+1, "MAPE"] = mape
    # print
    print("Step %02d, MAE: %.4f, RMSE: %.4f, MAPE:%.4f" % (i+1, mae, rmse, mape))

# save
np.save(output_path + '-prediction.npy', prediction)
np.save(output_path + '-target.npy', target)
evaluation_result.to_csv(output_path + '-evaluation.csv', float_format="%.4f")