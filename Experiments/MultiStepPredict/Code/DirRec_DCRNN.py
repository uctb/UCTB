import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import DCRNN
from UCTB.evaluation import metric
from UCTB.preprocess.GraphGenerator import GraphGenerator

class my_data_loader(NodeTrafficLoader):

    def __init__(self, **kwargs):

        super(my_data_loader, self).__init__(**kwargs) 
        
        # generate LM
        graph_obj = GraphGenerator(graph=kwargs['graph'], data_loader=self)
        self.AM = graph_obj.AM
        self.LM = graph_obj.LM

    def diffusion_matrix(self, filter_type='random_walk'):
        def calculate_random_walk_matrix(adjacent_mx):
            d = np.array(adjacent_mx.sum(1))
            d_inv = np.power(d, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = np.diag(d_inv)
            random_walk_mx = d_mat_inv.dot(adjacent_mx)
            return random_walk_mx
        assert len(self.AM) == 1

        diffusion_matrix = []
        if filter_type == "random_walk":
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0]).T)
        elif filter_type == "dual_random_walk":
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0]).T)
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0].T).T)
        return np.array(diffusion_matrix, dtype=np.float32)


# params
dataset_name = "Bike_NYC"
model_name = "DirRec_DCRNN"
output_path = "../Outputs/"+model_name+"-"+dataset_name
model_dir = "../Outputs/model_dir"
code_version = model_name+"-"+dataset_name
batch_size = 64
n_pred = 12
gpu_device = '0'

data_loader = my_data_loader(dataset='Bike', city='NYC', train_data_length='365',
                             closeness_len=6, period_len=7, trend_len=4, target_length=n_pred, graph='Correlation', normalize=True)

start = time.time()

diffusion_matrix = data_loader.diffusion_matrix()

# define n_pred model to train
model_list = []
temp_predict = []
temp_trainX = np.concatenate((data_loader.train_trend.transpose([0, 2, 1, 3]), data_loader.train_period.transpose([0, 2, 1, 3]), data_loader.train_closeness.transpose([0, 2, 1, 3])), axis=1)
for i in range(n_pred):
    temp_input_len = data_loader.closeness_len + data_loader.period_len + data_loader.trend_len + i
    temp_code_version = code_version + "-Step_"+ str(i+1)
    if i != 0:
        temp_trainX = np.concatenate((temp_trainX,temp_predict), axis=1)
    temp_model = DCRNN(num_nodes=data_loader.station_number,
        num_diffusion_matrix=diffusion_matrix.shape[0],
        num_rnn_units=64,
        num_rnn_layers=1,
        max_diffusion_step=2,
        seq_len=temp_input_len,
        use_curriculum_learning=False,
        input_dim=1,
        output_dim=1,
        cl_decay_steps=1000,
        target_len=1,
        lr=1e-4,
        epsilon=1e-3,
        optimizer_name='Adam',
        code_version=temp_code_version,
        model_dir=model_dir,
        gpu_device=gpu_device)
    # tf-graph
    temp_model.build()
    # training
    temp_model.fit(inputs=temp_trainX,
        diffusion_matrix=diffusion_matrix,
        target=data_loader.train_y[:, :, i].reshape([-1, 1, data_loader.station_number, 1]),
        batch_size=batch_size,
        sequence_length=data_loader.train_sequence_len,
        early_stop_length=100,
        max_epoch=1000)
    # save
    model_list.append(temp_model)
    # prediction
    temp_predict = temp_model.predict(
        inputs=temp_trainX,
        diffusion_matrix=diffusion_matrix,
        target=data_loader.train_y[:, :, i].reshape([-1, 1, data_loader.station_number, 1]),
        sequence_length=data_loader.train_sequence_len,
        output_names=['prediction']
    )
    # predict shape is [train_sequence_len, output_dim, station_number]
    temp_predict = temp_predict['prediction']
    temp_predict = temp_predict.reshape((temp_predict.shape[0], temp_predict.shape[1], temp_predict.shape[2], 1))

# use n_pred model to predict n_pred step
predict_list = []
temp_predict = []
temp_testX = np.concatenate((data_loader.test_trend.transpose([0, 2, 1, 3]), data_loader.test_period.transpose([0, 2, 1, 3]), data_loader.test_closeness.transpose([0, 2, 1, 3])), axis=1)
for i in range(n_pred):
    temp_model = model_list[i]
    if i != 0:
        temp_testX = np.concatenate((temp_testX, temp_predict), axis=1)
    temp_predict = temp_model.predict(
        inputs=temp_testX,
        diffusion_matrix=diffusion_matrix,
        target=data_loader.test_y[:, :, i].reshape([-1, 1, data_loader.station_number, 1]),
        sequence_length=data_loader.test_sequence_len,
        output_names=['prediction']
    )
    temp_predict = temp_predict['prediction']
    predict_list.append(temp_predict)
    temp_predict = temp_predict.reshape((temp_predict.shape[0], temp_predict.shape[1], temp_predict.shape[2], 1))

print('Total time cost is %.3f' % float(time.time()-start))

# Evaluation
predict_list = np.array(predict_list)
predict_list = predict_list.transpose([1, 3, 0, 2])
predict_list = predict_list.reshape((predict_list.shape[0], predict_list.shape[1], predict_list.shape[2]))
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