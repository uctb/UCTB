import nni

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
from UCTB.evaluation import metric

params = nni.get_next_parameter()

data_loader = NodeTrafficLoader(dataset=params['Dataset'], city=params['City'], with_lm=False, normalize=False, test_ratio=0.1)

test_start_index = data_loader.traffic_data.shape[0] - data_loader.test_data.shape[0]

val_start_index = data_loader.traffic_data.shape[0] - data_loader.test_data.shape[0] * 2

hm_obj = HM(c=int(params['CT']), p=int(params['PT']), t=int(params['TT']))

val_prediction = hm_obj.predict(val_start_index, data_loader.traffic_data[:test_start_index],
                                time_fitness=data_loader.dataset.time_fitness)

test_prediction = hm_obj.predict(test_start_index, data_loader.traffic_data, time_fitness=data_loader.dataset.time_fitness)

val_rmse = metric.rmse(val_prediction, data_loader.traffic_data[val_start_index: test_start_index])

test_rmse = metric.rmse(test_prediction, data_loader.test_data)

print(val_rmse, test_rmse)

nni.report_final_result({
    'default': val_rmse,
    'test-rmse': test_rmse,
})