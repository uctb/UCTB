from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian',
                                closeness_len=6, period_len=7, trend_len=4,
                                with_lm=False, with_tpe=False, normalize=False)

model = ARIMA(order=(6, 0, 2))
model.fit(data_loader)
results = model.predict(data_loader)
print('RMSE', metric.rmse(results, data_loader.test_y, threshold=0))
# or
model.eval()
