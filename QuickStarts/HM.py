from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
from UCTB.evaluation import metric
from UCTB.utils import save_predict_in_dataset

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=0, period_len=0, trend_len=4,
                                with_lm=False, normalize=False)

hm_obj = HM(c=data_loader.closeness_len,
            p=data_loader.period_len, t=data_loader.trend_len)

prediction = hm_obj.predict(closeness_feature=data_loader.test_closeness,
                            period_feature=data_loader.test_period,
                            trend_feature=data_loader.test_trend)

#save_predict_in_dataset(data_loader, prediction, "HM")

print('RMSE', metric.rmse(prediction, data_loader.test_y))
