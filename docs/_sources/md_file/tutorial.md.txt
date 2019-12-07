## Tutorial

#### Use datasets from UCTB (Not yet finished)

UCTB provides **bike traffic data of three cities: NYC, Chicago and DC**. A dataset api was provided to accuse these data.

For example if we want to use the bike data of NYC, we first initialize a data loader object

```python
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC')
```

Following are some examples to help you know more about the data:

Input:

```python
# traffic data 
print('Data time range', data_loader.dataset.time_range)
print('Traffic data shape:', data_loader.traffic_data.shape)
# The first dimension of data_loader.traffic_data is the length of time-sequence.
# the second dimension is the number of stations
print('Time fitness:', data_loader.dataset.time_fitness, 'minutes')
print('Time sequence length:', data_loader.traffic_data.shape[0])
print('Number of stations:', data_loader.traffic_data.shape[1])
```

Output:

```bash
Data time range ['2013-07-01', '2017-09-30']
Traffic data shape: (37248, 717)
Time fitness: 60 minutes
Time sequence length: 37248
Number of stations: 717
```

Input:

```python
import matplotlib.pyplot as plt
plt.plot(data_loader.traffic_data[:, 0])
```

Output:

<img src='src/image/toturial_p1_dataplot.png' width=50%>

#### Build your own datasets (Not yet finished)

A genereal data templeate is provided as following:

```python
release_data = {
    "TimeRange": ['YYYY-MM-DD', 'YYYY-MM-DD'],
    "TimeFitness": 60, # minutes
    
    "Node": {
        "TrafficNode": np.array, # with shape [time, num-of-node]
        "TrafficMonthlyInteraction": np.array, # with shape [month, num-of-node. num-of-node]
        "StationInfo": {id: [build-time, # Could alos be integer
                             lat, lng, name]},
        "POI": []
    },
	
    "Grid": {
        "TrafficGrid": [],
        "GridLatLng": [],
        "POI": []
    },

    "ExternalFeature": {
         "Weather": [time, weather-feature-dim]
    }
}
```

#### Use build-in models from UCTB

#### Build your own model using UCTB
