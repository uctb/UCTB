# Urban Datasets

UCTB is designed for urban computing in various scenarios. Currently, It releases [a public dataset repository](https://github.com/uctb/Urban-Dataset) including bike sharing, ride sharing, traffic speed, and pedestrian counting applications. **If you are interested in this project, making a contribution to the dataset is strongly welcomed :)**

## Open Datasets

Some description and external link of open datasets are listed in the table below.

| **Application**  | **City**  | **Time Range** | **Number of Station**| **Temporal Granularity** | **Dataset Link** |**Raw Data Link** |
| :--------------: | :-------: | :-------------: | :-------------: | :-------------: | :----------------------------------------------------------: |:----------------------------------------------------------: |
|   Bike-sharing   |    NYC    |  2013.07.01-2017.09.30    |820|5 minutes| [66.0M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/5_minutes/Bike_NYC.zip) | [source](https://www.citibikenyc.com/system-data)|
|   Bike-sharing   |  Chicago  |  2013.07.01-2017.09.30   |585|5 minutes| [30.2M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/5_minutes/Bike_Chicago.zip) | [source](https://www.divvybikes.com/system-data)|
|   Bike-sharing   |    DC     |  2013.07.01-2017.09.30    |532|5 minutes| [31.0M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/5_minutes/Bike_DC.zip) | [source](https://www.capitalbikeshare.com/system-data) |
| Pedestrian Count | Melbourne |   2021.01.01-2022.11.01    |55|60 minutes| [1.18M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Pedestrian/60_minutes/Pedestrian_Melbourne.pkl.zip) |[source](https://data.melbourne.vic.gov.au/pages/home/)|
|  Vehicle Speed   |    LA     |   2012.03.01-2012.06.28     |207|5 minutes| [11.8M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Speed/5_minutes/METR_LA.zip) |[source](https://github.com/liyaguang/DCRNN)|
|  Vehicle Speed   |    BAY    |   2017.01.01-2017.07.01     |325|5 minutes| [27.9M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Speed/5_minutes/PEMS_BAY.zip) |[source](https://github.com/liyaguang/DCRNN)|
|   Ride-sharing   |  Chicago  |  2013.01.01-2018.01.01 |121|60 minutes| [9.1M](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Taxi/15_minutes/Taxi_Chicago.zip) |[source](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)|

## Load UCTB Dataset

<!-- TODO: 介绍pickle -->
The `pickle` module is an external library that comes built-in with Python and provides functionality for converting Python objects into a byte stream (serialization) and restoring them back to their original state (deserialization). We use it to help data format instances to transform between memory and disk.

### Dataset format

Our abstract data format is externalized in the way of class `dict` in programming language Python as follows.

```python
# Let's say ``my_dataset`` is your dataset.
my_dataset = {
    "TimeRange": ['YYYY-MM-DD', 'YYYY-MM-DD'],
    "TimeFitness": 60, # Minutes
    
    "Node": {
        "TrafficNode": np.array, # With shape [time, num-of-node]
        "TrafficMonthlyInteraction": np.array, # With shape [month, num-of-node. num-of-node]
        "StationInfo": list # elements in it should be [id, build-time, lat, lng, name]
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

### Use datasets from Urban Datasets

In this section, we will introduce how to get the dataset from Urban_Dataset and read the dataset using python.

You are proposed to download the zip file from the [link](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Pedestrian/60_minutes/Pedestrian_Melbourne.pkl.zip) and unzip the file. Let's say the following scripts are placed at the same directory with the dataset.

```python
import pickle as pkl
import numpy as np
data_path = 'Pedestrian_Melbourne.pkl'
with open(data_path,'rb') as fp:
    data = pkl.load(fp)
```

Take a look at the necessary information about the dataset:

```python
# Traffic data 
print('Data time range', data['TimeRange'])
print('Traffic data shape:', np.shape(data['Node']['TrafficNode']))
# The first dimension of data['Node']['TrafficNode'] is the length of time-sequence.
# The second dimension is the number of stations.
print('Time fitness:', data['TimeFitness'], 'minutes')
print('Time sequence length:', data['Node']['TrafficNode'].shape[0])
print('Number of stations:', data['Node']['TrafficNode'].shape[1])
```

    Data time range ['2021-01-01', '2022-11-01']
    Traffic data shape: (16056, 55)
    Time fitness: 60 minutes
    Time sequence length: 16056
    Number of stations: 55

Visualize the distribution of the traffic data:

```python
import matplotlib.pyplot as plt
plt.plot(data['Node']['TrafficNode'][:, 0])
plt.show()
```

![png](src/image/toturial_p1_dataplot.png)

## Build your own datasets

If you want to apply uctb dataloaders to your dataset, make your dataset compatible with the template as section 3.2.1 shown. And then save it with package ``pickle`` to a local path ``pkl_file_name``.

```python
import pickle
pkl_file_name = './my_dataset.pkl'  
with open(pkl_file_name, 'wb') as handle:
    pickle.dump(my_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

Finally, you can make uses of your dataset by UCTB's loader APIs:

```python
data_loader = NodeTrafficLoader(dataset=pkl_file_name)
```

Also, we provide interface to help build your own dataset, through which we clarify whether a field is necessary or optional when building a UCTB dataset.

To build a UCTB dataset, it is necessary to provide variables listed as below.

|variable_name|description|
|:--|:--|
|time_fitness|The length of the interval between adjacent slots|
|time_range| the time interval at the beginning and end of the data |
|traffic_node| the spatio-temporal information |
|node_satation_info| the basic information of each data collecting node|
|dataset_name| name of the dataset |
|city| A variable used to integrate holiday and weather information to traffic data|

Then, use the specified path to save the dataset, otherwise it will be saved in the current run-time path.

Although it's diffcult to form an integrated function to include all situation you may meet during the transforming process, there are some procedures you might obey to simplify the data preprocessing.

- Data preprocessing
    1. Zero values
    2. Missing values(NA)
    3. Unknown values
    4. Abnormal values
    5. duplicates
    6. Statistics(station number and time slots)
- Dictionary building
    - Basic information(time range and time fitness)
    - Traffic node building
        - Spatio-temporal raster data building
            1. Initialization
            2. iterate raw data table and fill the matrix
        - Station information
    - Traffic grid building
    - External feature

Now, we assume that you have already finished variable preparation. UCTB provide API to assist you with dataset building.

```python
build_uctb_dataset(traffic_node=traffic_node, time_fitness=time_fitness, 
                node_station_info=node_station_info, time_range=time_range, 
                output_dir='tmp_dir', dataset_name='dataset', city = 'Chicago')
```

Also, if you want to check what fields are in your datasets, set the argument ``print_dataset`` to ``True``.

```python
build_uctb_dataset(traffic_node=traffic_node, time_fitness=time_fitness, 
                node_station_info=node_station_info, time_range=time_range, 
                output_dir='tmp_dir', dataset_name='dataset', city = 'Chicago', print_dataset=True)
```

Output:

    dataset[TimeRange]:<class 'list'>  (len=2)
    dataset[TimeFitness]:<class 'int'>
    dataset[Node]:<class 'dict'>{
        dataset[Node][TrafficNode]:<class 'numpy.ndarray'>  (shape=(37248, 532))
        dataset[Node][StationInfo]:<class 'list'>  (len=(532, 5))
        dataset[Node][TrafficMonthlyInteraction]:<class 'NoneType'>
    }
    dataset[Grid]:<class 'dict'>{
        dataset[Grid][TrafficGrid]:<class 'NoneType'>
        dataset[Grid][GridLatLng]:<class 'NoneType'>
    }
    dataset[ExternalFeature]:<class 'dict'>{
        dataset[ExternalFeature][Weather]:<class 'list'>  (len=0)
    }
    dataset[LenTimeSlots]:<class 'int'>

What's more, if you want to integrate additional information of the dataset, just specify the optional argument as bellow.

|variable_name|description|
|:--|:--|
|traffic_monthly_interaction| the interactive information among data collecting nodes. |
|poi| point of interests |
|traffic_grid| the spatio-temporal information in grid format. |
|gird_lat_lng| the basic information of each data collecting grid.|
|Weather| the weather information of each day. |

for example, specify the argument ``external_feature_weather`` with numpy.array object.

```python
build_uctb_dataset(traffic_node=traffic_node, time_fitness=time_fitness, 
                node_station_info=node_station_info, time_range=time_range, 
                output_dir='tmp_dir', dataset_name='dataset', city = 'Chicago', 
                print_dataset=True, external_feature_weather=np.zeros([37248,26]))
```

The code above use zero matrix to specify the argument ``external_feature_weather``. While in practical application scenario, you should substitute it with real feather matrix. The first dimension of the matrx is the number of time slots, and the second dimension corresponds to the dimension of weather features.