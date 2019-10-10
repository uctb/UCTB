from UCTB.model import HM
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC',
                                closeness_len=1, period_len=1, trend_len=2,
                                with_lm=False, normalize=False)

model = HM(c=1, p=1, t=2)
results = model.predict(data_loader)
print('RMSE', metric.rmse(results, data_loader.test_y, threshold=0))
# or
model.eval()
