## Tutorial

#### Use datasets from UCTB

UCTB provides **bike traffic data of three cities: NYC, Chicago and DC**. A dataset api was provided to accuse these data.

For example if we want to use the bike data of NYC, we first initialize a data loader object

```python
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC')
```



#### Build your own datasets

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



------

<u>[Back To HomePage](../index.html)</u>