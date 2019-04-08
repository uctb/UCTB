## Quick Start with HM (Historical Mean)

```python
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HM
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', with_lm=False)
start_index = data_loader.traffic_data.shape[0] - data_loader.test_data.shape[0]

# Build model
hm_obj = HM(d=7, h=0)

# Predict
prediction = hm_obj.predict(start_index, data_loader.traffic_data, time_fitness=data_loader.dataset.time_fitness)

print('RMSE', metric.rmse(prediction, data_loader.test_data, threshold=0))
```

## Quick Start with ARIMA

```python
import numpy as np
from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC')

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)

    try:
        
        # Train the ARIMA model
        model_obj = ARIMA(data_loader.train_data[:, i], [6, 0, 2])
        
        # Predict
        p = model_obj.predict(data_loader.test_x[:, :, i, 0])
        
    except Exception as e:
        print('Converge failed with error', e)
        print('Using zero as prediction')
        p = np.zeros([data_loader.test_x[:, :, i, 0].shape[0], 1])

    prediction.append(p)

    print(np.concatenate(prediction, axis=-1).shape)

prediction = np.expand_dims(np.concatenate(prediction, axis=-1), 2)

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))
```

## Quick Start with HMM

```python
import numpy as np
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import HMM
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', with_lm=False)

prediction = []

for station_index in range(data_loader.station_number):
    
    try:
        
        # Train the HMM model
        hmm = HMM(num_components=8, n_iter=1000)
        hmm.fit(data_loader.train_data[:, station_index:station_index+1])
        
        # Predict
        p = []
        for time_index in range(data_loader.test_x.shape[0]):
            p.append(hmm.predict(data_loader.test_x[time_index, :, station_index, :], length=1))
            
    except Exception as e:
        print('Failed at station', station_index, 'with error', e)
        p = [[0] for _ in range(data_loader.test_x.shape[0])]

    prediction.append(p)
    
prediction = np.transpose(prediction, (1, 0, 2))

print('RMSE', metric.rmse(prediction, data_loader.test_y, threshold=0))
```

## Quick Start with XGBoost

```python
import numpy as np
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import XGBoost
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', with_lm=False)

prediction = []

for i in range(data_loader.station_number):

    print('*************************************************************')
    print('Station', i)
    
    # Train the XGBoost model
    model = XGBoost(max_depth=10)
    model.fit(data_loader.train_x[:, :, i, 0], data_loader.train_y[:, i], num_boost_round=20)
	
    # Predict
    p = model.predict(data_loader.test_x[:, :, i, 0]).reshape([-1, 1])
    prediction.append(p)

prediction = np.concatenate(prediction, axis=-1)

print('RMSE', metric.rmse(prediction, data_loader.test_y.reshape([-1, data_loader.station_number]), threshold=0))
```

------

<u>[Back To HomePage](../index.html)</u>

