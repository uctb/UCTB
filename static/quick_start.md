## Quick Start with HM (Historical Mean)

```python
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=1, period_len=1, trend_len=2,
                                with_lm=False, normalize=False)

hm_obj = HM(c=data_loader.closeness_len, p=data_loader.period_len, t=data_loader.trend_len)

prediction = hm_obj.predict(closeness_feature=data_loader.test_closeness,
                            period_feature=data_loader.test_period,
                            trend_feature=data_loader.test_trend)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))
```

## Quick Start with ARIMA

```python
import numpy as np

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric


data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=24, period_len=0, trend_len=0,
                                with_lm=False, normalize=False)

test_prediction_collector = []
for i in range(data_loader.station_number):
    try:
        model_obj = ARIMA(time_sequence=data_loader.train_closeness[:, i, -1, 0],
                          order=[6, 0, 1], seasonal_order=[0, 0, 0, 0])
        test_prediction = model_obj.predict(time_sequences=data_loader.test_closeness[:, i, :, 0],
                                            forecast_step=1)
    except Exception as e:
        print('Converge failed with error', e)
        print('Using last as prediction')
        test_prediction = data_loader.test_closeness[:, i, -1:, :]
    test_prediction_collector.append(test_prediction)
    print('Station', i, 'finished')

test_rmse = metric.rmse(np.concatenate(test_prediction_collector, axis=-2), data_loader.test_y, threshold=0)

print('test_rmse', test_rmse)
```

## Quick Start with HMM

```python
import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HMM
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC',
                                closeness_len=12, period_len=0, trend_len=0,
                                with_lm=False, normalize=False)

prediction = []
for station_index in range(data_loader.station_number):
    # train the hmm model
    try:
        hmm = HMM(num_components=8, n_iter=100)
        hmm.fit(data_loader.train_closeness[:, station_index:station_index+1, -1, 0])
        # predict
        p = []
        for time_index in range(data_loader.test_closeness.shape[0]):
            p.append(hmm.predict(data_loader.test_closeness[time_index, station_index, :, :], length=1))
    except Exception as e:
        print('Failed at station', station_index, 'with error', e)
        # using zero as prediction
        p = [[[0]] for _ in range(data_loader.test_closeness.shape[0])]

    prediction.append(np.array(p)[:, :, 0])
    print('Node', station_index, 'finished')

prediction = np.array(prediction).transpose([1, 0, 2])
print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))
```

## Quick Start with XGBoost

```python
import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric

data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', closeness_len=6, period_len=7, trend_len=4,
                                with_lm=False, normalize=False)

prediction_test = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    model = XGBoost(n_estimators=100, max_depth=3, objective='reg:squarederror')

    model.fit(np.concatenate((data_loader.train_closeness[:, i, :, 0],
                              data_loader.train_period[:, i, :, 0],
                              data_loader.train_trend[:, i, :, 0],), axis=-1),
              data_loader.train_y[:, i, 0])

    p_test = model.predict(np.concatenate((data_loader.test_closeness[:, i, :, 0],
                                           data_loader.test_period[:, i, :, 0],
                                           data_loader.test_trend[:, i, :, 0],), axis=-1))

    prediction_test.append(p_test.reshape([-1, 1, 1]))

prediction_test = np.concatenate(prediction_test, axis=-2)

print('Test RMSE', metric.rmse(prediction_test, data_loader.test_y, threshold=0))
```

------

<u>[Back To HomePage](../index.html)</u>

