import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric


# params
dataset_name = "Bike_NYC"
model_name = "DirRec_ARIMA"
output_path = "../Outputs/"+model_name+"-"+dataset_name

n_pred = 12

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=24, period_len=0, trend_len=0, target_length=n_pred, with_lm=False, normalize=False)

start = time.time()

test_prediction_collector = []
for i in range(data_loader.station_number):
    try:
        model_obj = ARIMA(time_sequence=data_loader.train_closeness[:, i, -1, 0],
                          order=[6, 0, 1], seasonal_order=[0, 0, 0, 0])
        test_prediction = model_obj.predict(time_sequences=data_loader.test_closeness[:, i, :, 0],
                                            forecast_step=n_pred)
    except Exception as e:
        print('Converge failed with error', e)
        print('Using last as prediction')
        test_prediction = data_loader.test_closeness[:, i, -1:, :]
    test_prediction_collector.append(test_prediction)
    print('Station', i, 'finished')

predict_list = np.array(test_prediction_collector)
predict_list = predict_list.transpose([1, 0, 2])

print('Total time cost is %.3f' % float(time.time()-start))
prediction = predict_list
prediction = np.where(prediction>0, prediction, 0)
target = data_loader.test_y
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