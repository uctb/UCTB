import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import STMeta
from UCTB.evaluation import metric
from UCTB.preprocess.GraphGenerator import GraphGenerator


# params
dataset_name = "Bike_NYC"
model_name = "DirRec_STMeta"
output_path = "../Outputs/"+model_name+"-"+dataset_name
model_dir = "../Outputs/model_dir"
code_version = model_name+"-"+dataset_name
n_pred = 12
gpu_device = '1'

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=6, period_len=7, trend_len=4, target_length=n_pred, normalize=True)

start = time.time()

# Build Graph
graph_obj = GraphGenerator(graph='Correlation', data_loader=data_loader)

# define n_pred model to train
model_list = []
temp_predict = []
temp_closeness_feature = data_loader.train_closeness
for i in range(n_pred):
    temp_closeness_len = data_loader.closeness_len + i
    temp_code_version = code_version + "-Step_"+ str(i+1)
    if i != 0:
        temp_closeness_feature = np.concatenate((temp_closeness_feature, temp_predict), axis=2)
    temp_model = STMeta(closeness_len=temp_closeness_len,
                    period_len=data_loader.period_len,
                    trend_len=data_loader.trend_len,
                    num_node=data_loader.station_number,
                    num_graph=graph_obj.LM.shape[0],
                    external_dim=data_loader.external_dim,
                    code_version=temp_code_version,
                    model_dir=model_dir,
                    gpu_device=gpu_device)
    # Build tf-graph
    temp_model.build()
    # Training
    temp_model.fit(closeness_feature=temp_closeness_feature,
                period_feature=data_loader.train_period,
                trend_feature=data_loader.train_trend,
                laplace_matrix=graph_obj.LM,
                target=data_loader.train_y[:,:,i].reshape((-1, data_loader.station_number, 1)),
                external_feature=data_loader.train_ef,
                sequence_length=data_loader.train_sequence_len,
                auto_load_model = False)
    # save
    model_list.append(temp_model)
    # prediction
    temp_predict = temp_model.predict(closeness_feature=temp_closeness_feature,
                period_feature=data_loader.train_period,
                trend_feature=data_loader.train_trend,
                laplace_matrix=graph_obj.LM,
                target=data_loader.train_y[:,:,i].reshape((-1, data_loader.station_number, 1)),
                external_feature=data_loader.train_ef,
                output_names=['prediction'],
                sequence_length=data_loader.train_sequence_len)
    # predict shape is [sequence_len, station_num, 1]
    temp_predict = temp_predict['prediction']
    temp_predict = temp_predict.reshape((temp_predict.shape[0], temp_predict.shape[1], 1, 1))

# use n_pred model to predict n_pred step
predict_list = []
temp_predict = []
temp_closeness_feature = data_loader.test_closeness
for i in range(n_pred):
    temp_model = model_list[i]
    if i != 0:
        temp_closeness_feature = np.concatenate((temp_closeness_feature, temp_predict), axis=2)
    # prediction
    temp_predict = temp_model.predict(closeness_feature=temp_closeness_feature,
                period_feature=data_loader.test_period,
                trend_feature=data_loader.test_trend,
                laplace_matrix=graph_obj.LM,
                target=data_loader.test_y[:,:,i].reshape((-1, data_loader.station_number, 1)),
                external_feature=data_loader.test_ef,
                output_names=['prediction'],
                sequence_length=data_loader.test_sequence_len)
    temp_predict = temp_predict['prediction']
    # predict shape is [sequence_len, station_num, 1]
    predict_list.append(temp_predict)
    temp_predict = temp_predict.reshape((temp_predict.shape[0], temp_predict.shape[1], 1, 1))

print('Total time cost is %.3f' % float(time.time()-start))

# Evaluation
predict_list = np.concatenate(predict_list, axis=2)
prediction = data_loader.normalizer.min_max_denormal(predict_list)
prediction = np.where(prediction>0, prediction, 0)
target = data_loader.normalizer.min_max_denormal(data_loader.test_y)

evaluation_result = pd.DataFrame(columns=["MAE", "RMSE", "MAPE"], index=range(1, n_pred+1))
for i in range(n_pred):
    # reshape
    cur_prediction = prediction[:,:,i]
    cur_target = target[:,:,i]
    # result
    mae = MAE(cur_prediction, cur_target)
    rmse = metric.rmse(cur_prediction, cur_target, threshold=0)
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
    
