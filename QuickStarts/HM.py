from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Metro', city='Shanghai', with_lm=False, normalize=False)

start_index = data_loader.traffic_data.shape[0] - data_loader.test_data.shape[0]

hm_obj = HM(c=0, p=0, t=4)

prediction = hm_obj.predict(start_index, data_loader.traffic_data, time_fitness=data_loader.dataset.time_fitness)

print('RMSE', metric.rmse(prediction, data_loader.test_data, threshold=0))