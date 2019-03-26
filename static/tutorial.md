<link rel="stylesheet" href="./static/css/misty.css" type="text/css"/>

## Tutorial

- ##### Use datasets from UCTB

- ##### Build your own datasets

Data template

```python
release_data = {
    "TimeRange": ['2013-07-01', '2017-09-30'],
    "TimeFitness": 60, # minutes
    
    "Node": {
        "TrafficNode": np.array, # [time, num-of-node]
        "TrafficMonthlyInteraction": np.array, # [month, num-of-node. num-of-node]
        "StationInfo": {id: [build-time, lat, lng, name]},
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

- ##### Use build-in models from UCTB

- ##### Build your own model using UCTB



------

<u>[Back To HomePage](../index.html)</u>