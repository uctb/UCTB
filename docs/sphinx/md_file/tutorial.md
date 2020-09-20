
## Tutorial

#### Use datasets from UCTB

UCTB is designed for urban computing in various scenarios. Currently, It presets a public dataset about bikesharing. This dataset was collected from U.S. open data portals, including 49 million, 13 million, and 14 million historical flow records in [New York City](https://www.citibikenyc.com/system-data) (``NYC``), [Chicago](https://www.divvybikes.com/system-data) (``Chicago``) and [Washington, D.C](https://www.capitalbikeshare.com/system-data) (``DC``), respectively. Each record contains the start station, start time, stop station, stop time, etc. We predict the number of bikesharing demands in each station (i.e., the number of bike borrowers).

In the future version, we consider releasing more datasets covering other applications such as ridesharing, metro traffic flow, and electrical charging station usage. **If you are interested in this project, making a contribution to the dataset is strongly welcomed :)** 

To help better accuse dataset, UCTB provides data loader APIs ``UCTB.dataset.data_loader``, which can be used to preprocess data, including data division, normalization, and extract temporal and spatial knowledge. 

In the following tutorial, we illustrate how to use ``UCTB.dataset.data_loader`` APIs to inspect the bikesharing dataset.


```python
from UCTB.dataset.data_loader import NodeTrafficLoader
```

We use 10% (``data_range=0.1``) of the bike data in New York as an example. Firstly, let's initialize a ``NodeTrafficLoader`` object:


```python
data_loader = NodeTrafficLoader(data_range=0.1, dataset='Bike', city='NYC')
```

Take a look at the necessary information about the dataset:


```python
# Traffic data 
print('Data time range', data_loader.dataset.time_range)
print('Traffic data shape:', data_loader.traffic_data.shape)
# The first dimension of data_loader.traffic_data is the length of time-sequence.
# The second dimension is the number of stations.
print('Time fitness:', data_loader.dataset.time_fitness, 'minutes')
print('Time sequence length:', data_loader.traffic_data.shape[0])
print('Number of stations:', data_loader.traffic_data.shape[1])
```

    Data time range ['2013-07-01', '2017-09-30']
    Traffic data shape: (3724, 717)
    Time fitness: 60 minutes
    Time sequence length: 3724
    Number of stations: 717


Visualize the distribution of the traffic data:


```python
import matplotlib.pyplot as plt
plt.plot(data_loader.traffic_data[:, 0])
plt.show()
```


![png](https://uctb.github.io/UCTB/sphinx/md_file/src/image/toturial_p1_dataplot.png)

#### Build your own datasets

To make loader APIs compatible with your own data, you can store it in a ``dict`` variable with formats as follows.


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

And then save it with package ``pickle`` to a local path ``pkl_file_name``.


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

For better understanding how to build dataset used in UCTB from the original dataset, here are two examples:

1. Regional trajectory data for Xi'an and Chengdu.  [Guide](./static/MakeDatasetDiDi.html)

2. [TTI](https://github.com/didi/TrafficIndex) data for Chengdu, Jinan, Haikou, Shenzhen, Suzhou and Xi'an.  [Guide](./static/MakeDatasetDiDi_TTI.html)

The above data comes from [https://gaia.didichuxing.com]( https://outreach.didichuxing.com/research/opendata/en/ ), and we are especially grateful for the data provided by the DiDi Chuxing GAIA Initiative.

#### Use build-in models from UCTB

##### Use single temporal feature in regression

UCTB provides many classical and popular spatial-temporal predicting models. These models can be used to either predicting series for a single station or all stations. You can find the details in [``UCTB.model``](./static/current_supported_models.html).

The following example shows how to use a **Hidden Markov model (HMM)** to handle a simple time series predicting a problem. We will try to predict the bike demands ``test_y`` of a fixed station ``target_node`` in New York City by checking back the historical demands in recent time slots ``train_closeness``.


```python
import numpy as np
import matplotlib.pyplot as plt

from UCTB.model import HMM
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

target_node = 233
```

When initializing the loader, we use past ``12`` time slots (timesteps) of closeness as input, ``1`` timestep in the next as output and set the timesteps of other features ``period_len``, ``period_len`` to zero. 


```python
data_loader = NodeTrafficLoader(data_range=0.1, dataset='Bike', city='NYC',
                                closeness_len=12, period_len=0, trend_len=0,
                                target_length=1, test_ratio=0.2, 
                                normalize=False, with_lm=False, with_tpe=False)
```

The well-loaded data contain all ``717`` stations' data. Therefore it is needed to specify the target station by ``target_station``.

```python
print(data_loader.train_closeness.shape)
print(data_loader.test_closeness.shape)
print(data_loader.test_y.shape)
```

```python
(2967, 717, 12, 1)
(745, 717, 12, 1)
(745, 717, 1)
```


```python
train_x, test_x = data_loader.train_closeness[:, target_node:target_node+1, -1, 0], data_loader.test_closeness[:, target_node, :, :]
test_y = data_loader.test_y[:, target_node, 0]
```

Inspect the shape of data. Here are the all we need for one-station prediction.


```python
print(train_x.shape)
print(test_x.shape)
print(test_y.shape)
```

    (2967, 1)
    (745, 12, 1)
    (745,)


Build the HMM model.


```python
model = HMM(num_components=8, n_iter=50)
```

Now, we can fit the model with the train dataset.


```python
model.fit(x=train_x)
```

    Status: converged


When the model is converged, we make predictions on test data.


```python
predictions = []
for t in range(test_x.shape[0]):
    p = np.squeeze(model.predict(x=test_x[t], length=1))
    predictions.append(p)
```

We can evaluate the performance of the model by build-in ``UCTB.evaluation`` APIs.


```python
test_rmse = metric.rmse(predictions, test_y, threshold=0)
print(test_rmse)
```


    3.76137200105079

##### Use multiple temporal features in regression

In this case, let's take more temporal knowledge related to ``target_node`` into account. We will concatenate factors including ``closeness``, ``period``, and ``trend``, and use **XGBoost** as the predicting model.


```python
import numpy as np
import matplotlib.pyplot as plt

from UCTB.model import XGBoost
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric

target_node = 233

data_loader = NodeTrafficLoader(data_range=0.1, dataset='Bike', city='NYC',
                                closeness_len=6, period_len=7, trend_len=4,
                                target_length=1, test_ratio=0.2, 
                                normalize=False, with_lm=False, with_tpe=False)

train_closeness = data_loader.train_closeness[:, target_node, :, 0]
train_period = data_loader.train_period[:, target_nodze, :, 0]
train_trend = data_loader.train_trend[:, target_node, :, 0]
train_y = data_loader.train_y[:, target_node, 0]

test_closeness = data_loader.test_closeness[:, target_node, :, 0]
test_period = data_loader.test_period[:, target_node, :, 0]
test_trend = data_loader.test_trend[:, target_node, :, 0]
test_y = data_loader.test_y[:, target_node, 0]

train_X = np.concatenate([train_closeness, train_period, train_trend], axis=-1)
test_X = np.concatenate([test_closeness, test_period, test_trend], axis=-1)

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

model = XGBoost(n_estimators=100, max_depth=3, objective='reg:linear')
model.fit(train_X, train_y)
predictions = model.predict(test_X)
print('Test RMSE', metric.rmse(predictions, test_y, threshold=0))
```

    (2307, 17)
    (2307,)
    (745, 17)
    (745,)
    Test RMSE 3.3267457

#### Build your own model using UCTB

UCTB provides extendable APIs to build your own model. Currently, it can support the running of all the ``1.x`` version of **Tensorflow-based** models. In the following tutorial, we will show you how to takes the least efforts to implement a UCTB model.

Commonly, a new model needs to inherit ``BaseModel`` to acquire the features provided by UCTB, such as batch division, early stopping, etc. The necessary components for a subclass of ``BaseModel`` include:

- ``self.__init__()``. Define the model's parameters related to the architecture. You should call the super class's constructor at first.
- ``self.build()``. Build the architecture here. You should construct the graph at the beginning of this function and call the super class's ``build()`` function at the end.
- ``self._input``. The ``dict`` used to record the acceptable inputs of the model, whose keys are the parameter names in ``model.fit()`` and ``model.predict()`` and values are the name of related tensors.
- ``self._output``. The ``dict`` used to record the outputs of the model. You should fill the required keys ``prediction`` and ``loss`` with the names of tensors in your case.
- ``self._op``. The ``dict`` used to define all the operations for the model. Basic usage for it is to record the **training operation**, for example, the minimizing loss operation of an optimizer. Use key ``train_op`` to record it.

For more examples, you can refer to the implementations of build-in models in [``UCTB.model``](../UCTB.model.html#uctb-model-package).


```python
from UCTB.model_unit import BaseModel

class MyModel(BaseModel):
    def __init__(self,
                 
                 code_version='0',
                 model_dir='my_model',
                 gpu_device='0',
                ):
        super(MyModel, self).__init__(code_version=code_version, 
                                      model_dir=model_dir, gpu_device=gpu_device)
        ...
        
    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():
            ...
            self._input['inputs'] = inputs.name
            self._input['targets'] = targets.name
            
            ...
            self._output['prediction'] = predictions.name
            self._output['loss'] = loss.name
            self._op['train_op'] = train_op.name
            
        super(MyModel, self).build(init_vars=init_vars, max_to_keep=5) 
```

Next, in a concrete case, we will realize a **Long short-term memory (LSTM)** model to make the all-station prediction that accepts time series of `717` stations and predict the future of them as a whole. 

For the mechanism of LSTM, you can refer to 
[Gers, F. A., Schmidhuber, J., & Cummins, F. (1999). Learning to forget: Continual prediction with LSTM](https://www.researchgate.net/profile/Felix_Gers/publication/12292425_Learning_to_Forget_Continual_Prediction_with_LSTM/links/5759414608ae9a9c954e84c5/Learning-to-Forget-Continual-Prediction-with-LSTM.pdf).


```python
import numpy as np
import tensorflow as tf
from UCTB.dataset import NodeTrafficLoader
from UCTB.model_unit import BaseModel
from UCTB.preprocess import SplitData
from UCTB.evaluation import metric
```


```python
class LSTM(BaseModel):
    def __init__(self,
                 num_stations, 
                 num_layers, 
                 num_units, 
                 input_steps, 
                 input_dim,
                 output_steps,
                 output_dim,
                 code_version='0',
                 model_dir='my_lstm',
                 gpu_device='0'):
        super(LSTM, self).__init__(code_version=code_version, 
                                   model_dir=model_dir, gpu_device=gpu_device)
        self.num_stations = num_stations
        self.num_layers = num_layers
        self.num_units = num_units
        self.input_steps = input_steps
        self.input_dim = input_dim
        self.output_steps = output_steps
        self.output_dim = output_dim
        
    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():
            inputs = tf.placeholder(tf.float32, shape=(None, self.num_stations, 
                                                       self.input_steps, self.input_dim))
            targets = tf.placeholder(tf.float32, shape=(None, self.num_stations,
                                                       self.output_steps, self.output_dim))
            # record the inputs of the model
            self._input['inputs'] = inputs.name
            self._input['targets'] = targets.name

            inputs = tf.reshape(inputs, (-1, self.input_steps, self.num_stations*self.input_dim))
            
            def get_a_cell(num_units):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
                return lstm
            
            stacked_cells = tf.contrib.rnn.MultiRNNCell([get_a_cell(self.num_units) for _ in range(self.num_layers)], state_is_tuple=True)
            outputs, final_state = tf.nn.dynamic_rnn(stacked_cells, inputs, dtype=tf.float32)
            
            stacked_outputs = tf.reshape(outputs, shape=(-1, self.num_units*self.input_steps))
            predictions = tf.layers.dense(stacked_outputs, self.output_steps*self.num_stations*self.output_dim)
            predictions = tf.reshape(predictions, shape=(-1, self.num_stations, self.output_steps, self.output_dim))
            
            loss = tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))
            train_op = tf.train.AdamOptimizer().minimize(loss)
            
            # record the outputs and the operation of the model
            self._output['prediction'] = predictions.name
            self._output['loss'] = loss.name
            self._op['train_op'] = train_op.name
        
        # must call super class' function to build 
        super(LSTM, self).build(init_vars=init_vars, max_to_keep=5) 
```

Load the dataset by loader and transform them into the formats your model accepts. If the loader APIs are not filled your demands, you can inherit loader and wrapper it according to your desires (see [Quickstart](./quickstart.html) for more details).


```python
data_loader = NodeTrafficLoader(data_range=0.1, dataset='Bike', city='NYC',
                                closeness_len=6, period_len=0, trend_len=0,
                                target_length=1, test_ratio=0.2, 
                                normalize=True, with_lm=False, with_tpe=False)
train_y = np.expand_dims(data_loader.train_y, axis=-1)
test_y = np.expand_dims(data_loader.test_y, axis=-1)
```


```python
model = LSTM(num_stations=data_loader.station_number, 
             num_layers=2,
             num_units=512, 
             input_steps=6, 
             input_dim=1, 
             output_steps=1, 
             output_dim=1)
```


```python
model.build()
print(model.trainable_vars)  # count the trainble parameters
```

    6821581

Use your model to training and predicting. ``model.fit()`` method presets lots of useful functions, such as batch division and early stopping. Check them in [``UCTB.model_unit.BaseModel.BaseModel.fit``](../UCTB.model_unit.html#UCTB.model_unit.BaseModel.BaseModel.fit).


```python
model.fit(inputs=data_loader.train_closeness,
          targets=train_y,
          max_epoch=10,
          batch_size=64,
          sequence_length=data_loader.train_sequence_len,
          validate_ratio=0.1)
```

    No model found, start training
    Running Operation ('train_op',)
    Epoch 0: train_loss 0.016053785 val_loss 0.01606118
    Epoch 1: train_loss 0.015499311 val_loss 0.015820855
    Epoch 2: train_loss 0.015298592 val_loss 0.015657894
    Epoch 3: train_loss 0.015163456 val_loss 0.015559187
    Epoch 4: train_loss 0.015066812 val_loss 0.015342651
    Epoch 5: train_loss 0.015016247 val_loss 0.015287879
    Epoch 6: train_loss 0.014899823 val_loss 0.015249459
    Epoch 7: train_loss 0.014773054 val_loss 0.015098239
    Epoch 8: train_loss 0.014655286 val_loss 0.015097916
    Epoch 9: train_loss 0.014558283 val_loss 0.015108417

```python
predictions = model.predict(inputs=data_loader.test_closeness, 
                            sequence_length=data_loader.test_sequence_len)
```

Reverse the normalization by ``data_loader`` and evaluate the results:


```python
predictions = data_loader.normalizer.min_max_denormal(predictions['prediction'])
targets = data_loader.normalizer.min_max_denormal(test_y)
print('Test result', metric.rmse(prediction=predictions, target=targets, threshold=0))
```

    Test result 2.9765626570592545

Since we only use a short period of the dataset (``data_range=0.1``) in this toy example, the result looks good compared with the [experiment](./all_results.html#results-on-bike). You can also take a try to test the completed dataset on your model. 

#### Build your own graph with STMeta

Next, we will use the Top-K graph as an example to illustrate how to build customized graphs in UCTB. All of the code in this section can be found [here](https://anonymous.4open.science/r/561305b5-e65e-46c6-9371-ae76b85109ee/Experiments/CustomizedDemo/).

##### Top-K graph

First of all, the customized graphs used in this section is called Top-K graph. We construct the corresponding adjacent graph by marking the point pair that consist of each point and its nearest K points as 1, and the others are marked as 0. Then, we use the adjacent graph to generate the laplacian matrix for input. The hyperparameter K is designed via ad-hoc heuristics. In this demonstration, we chose 23 as the value of K.

##### Realize TopK graph analysis module

To adopt customized graphs (***e.g.,*** Top-K) in UCTB, you should first build your own analysis class by inheriting `UCTB.preprocess.GraphGenerator class`.

It is worth noting that the ultimate goal is to generate the member variables: `self.LM` and `self.AM`, which is the input matrix of the graph. In this phase, we need to make the corresponding analytical implementation according to the type of the custom graph passed in.

 ```python
# "UCTB/preprocess/topKGraph.py"
import heapq
import numpy as np
from UCTB.preprocess.GraphGenerator import GraphGenerator

# Define the class: topKGraph
class topKGraph(GraphGenerator):  # Init NodeTrafficLoader

    def __init__(self,**kwargs):

        super(topKGraph, self).__init__(**kwargs)
        
        for graph_name in kwargs['graph'].split('-'):

# As the basic graph is implemented in GraphGenerator, you only need to implement your own graph function instead of the existing one.
            if graph_name.lower() == 'topk':
                lat_lng_list = np.array([[float(e1) for e1 in e[2:4]]
                                         for e in self.dataset.node_station_info])
                # Handling
                AM = self.neighbour_adjacent(lat_lng_list[self.traffic_data_index],
                                        threshold=int(kwargs['threshold_neighbour']))
                LM = self.adjacent_to_laplacian(AM)

                if self.AM.shape[0] == 0:  # Make AM
                    self.AM = np.array([AM], dtype=np.float32)
                else:
                    self.AM = np.vstack((self.AM, (AM[np.newaxis, :])))

                if self.LM.shape[0] == 0:  # Make LM
                    self.LM = np.array([LM], dtype=np.float32)
                else:
                    self.LM = np.vstack((self.LM, (LM[np.newaxis, :])))

# Implement the details of building the Top-K graph.
    def neighbour_adjacent(self, lat_lng_list, threshold):
        adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
        for i in range(len(lat_lng_list)):
            for j in range(len(lat_lng_list)):
                adjacent_matrix[i][j] = self.haversine(
                    lat_lng_list[i][0], lat_lng_list[i][1], lat_lng_list[j][0], lat_lng_list[j][1])
        dis_matrix = adjacent_matrix.astype(np.float32)

        for i in range(len(dis_matrix)):
            ind = heapq.nlargest(threshold, range(len(dis_matrix[i])), dis_matrix[i].take)
            dis_matrix[i] = np.array([0 for _ in range(len(dis_matrix[i]))])
            dis_matrix[i][ind] = 1
        adjacent_matrix = (adjacent_matrix == 1).astype(np.float32)
        return adjacent_matrix
 ```

##### Redefine the call statement of the above class

```python
# "UCTB/Experiments/CustomizedDemo/STMeta_Obj_topk.py"

# Import the Class: topKGraph
from topKGraph import topKGraph
# Call topKGraph to initialize and generate AM and LM
graphBuilder = topKGraph(graph=args['graph'],
                         data_loader=data_loader,
                         threshold_distance=args['threshold_distance'],
                         threshold_correlation=args['threshold_correlation'],
                         threshold_interaction=args['threshold_interaction'],
                         threshold_neighbour=args['threshold_neighbour'])
# ......
```

##### Modify the function call location

Add the new graph name when fitting model and then execute it for experiments. [code](https://github.com/uctb/UCTB/blob/master/Experiments/CustomizedDemo/Runner_topk.py)

```python
os.system('python STMeta_Obj_topk.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p graph:Distance-Correlation-Line-TopK,MergeIndex:12')
```

We conduct experiments on `Metro_Shanghai` dataset and use the [STMeta_V1](https://uctb.github.io/UCTB/md_file/all_results.html#stmeta-version) to model both "Distance-Correlation-Line" graph and "Distance-Correlation-Line-TopK" and the results are following:

| **Metro: Shanghai** |             Graph              | Test-RMSE |
| :-----------------: | :----------------------------: | :-------: |
|      STMeta_V1      |   Distance-Correlation-Line    |  153.17   |
|      STMeta_V1      | Distance-Correlation-Line-TopK |  140.82   |

The results show that the performance of STMeta_V1 with the graph "Distance-Correlation-Line-TopK" is better than "Distance-Correlation-Line" model and the RMSE is reduced by about 12.4%, which validates the effectiveness of the topk graph for spatiotemporal modeling STMeta algorithm.

------

<u>[Back To HomePage](../index.html)</u>

