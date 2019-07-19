from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Metro', city='Chongqing',
                                closeness_len=1, period_len=1, trend_len=2,
                                with_lm=False, normalize=False)

hm_obj = HM(c=data_loader.closeness_len, p=data_loader.period_len, t=data_loader.trend_len)

prediction = hm_obj.predict(closeness_feature=data_loader.test_closeness,
                            period_feature=data_loader.test_period,
                            trend_feature=data_loader.test_trend)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))