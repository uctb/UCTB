# [Urban Computing ToolBox](https://github.com/Di-Chai/UCTB)

## Introduction

**Urban Computing ToolBox** is a package providing spatial-temporal predicting models. It contains both conventional models and state-of-art models. 

Currently the package supported the following models: ([Details](./static/current_supported_models.html))

- HM
- HMM
- ARIMA
- XGBoost
- DeepST
- ST-ResNet
- DCRNN
- STMeta

UCTB is a flexible and open package. You can use the data we provided or use your own data, the data structure is well stated in the tutorial chapter. You can build your own model based on model-units we provided and use the model-training class to train the model.

## Installation

### Install UCTB

##### Step 1: Install TensorFlow

You can skip to step 2 if you already installed tensorflow.

You can refer to this page <https://www.tensorflow.org/install> to install tensorflow, if you have a Nvidia GPU installed on you computer, we highly recommend you to install GPU version of tensorflow.

##### Step 2: Install UCTB

```bash
pip install --upgrade UCTB
```

The following required package will be installed or upgraded with UCTB:

```bash
'hmmlearn',
'keras',
'GPUtil',
'numpy',
'pandas',
'python-dateutil',
'scikit-learn',
'scipy',
'statsmodels',
'wget',
'xgboost',
'nni',
'chinesecalendar',
'PyYAML'
```

### UCTB Docker

You can also  use UCTB by docker. First pull uctb docker from docker hub.

```bash
docker pull dichai/uctb:v0.2.0
```

And  you then can run it.

```bash
docker run  --runtime=nvidia  -it -d dichai/uctb:v0.2.0 /bin/bash
```

## Quick start

- ##### Quick start with STMeta

```python
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import STMeta
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', graph='Correlation',
                                closeness_len=6, period_len=7, trend_len=4, normalize=True)

# Init model object
STMeta_Obj = STMeta(closeness_len=data_loader.closeness_len,
                                 period_len=data_loader.period_len,
                                 trend_len=data_loader.trend_len,
                                 num_node=data_loader.station_number,
                                 num_graph=data_loader.LM.shape[0])

# Build tf-graph
STMeta_Obj.build()
# Training
STMeta_Obj.fit(closeness_feature=data_loader.train_closeness,
                      period_feature=data_loader.train_period,
                      trend_feature=data_loader.train_trend,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      sequence_length=data_loader.train_sequence_len)

# Predict
prediction = STMeta_Obj.predict(closeness_feature=data_loader.test_closeness,
                                       period_feature=data_loader.test_period,
                                       trend_feature=data_loader.test_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       output_names=['prediction'],
                                       sequence_length=data_loader.test_sequence_len)

# Evaluate
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0))
```

- [Quick start with other models](./static/quick_start.html)

## Tutorial

- Compare different version of STMeta

- ##### [Use datasets from UCTB](./static/tutorial.html)

- ##### [Build your own datasets](./static/tutorial.html)

- ##### Use build-in models from UCTB


- ##### Build your own model using UCTB


## API Reference

- ##### Model Class


- ##### ModelUnit Class


- ##### Evaluation Class


- ##### Training Class


## Full examples

- ##### [Experiments on bike traffic-flow prediction](./static/experiment_on_bike.html)

- ##### [Experiments on metro traffic-flow prediction](./static/experiment_on_metro.html)

- ##### [Experiments on didi order demand prediction](./static/experiment_on_didi.html)

- Experiments on charge station demand prediction

## Contribute to UCTB (Delay)

- ##### Contribute Data
- ##### Contribute Models
- ##### Contribute Model-Units
