from UCTB.model import HMM
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian',
                                closeness_len=6, period_len=7, trend_len=4,
                                with_lm=False, with_tpe=False, normalize=False)

model = HMM(num_components=8, n_iter=1000)
model.fit(data_loader)
results = model.predict(data_loader)
print('RMSE', metric.rmse(results, data_loader.test_y, threshold=0))
# or
model.eval()
