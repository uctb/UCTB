from UCTB.model import XGBoost
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian',
                                closeness_len=12, period_len=0, trend_len=2,
                                with_lm=False, with_tpe=False, normalize=False)

model = XGBoost(n_estimators=10, max_depth=5)
model.fit(data_loader)
results = model.predict(data_loader)
print('RMSE', metric.rmse(results, data_loader.test_y, threshold=0))
# or
model.eval()