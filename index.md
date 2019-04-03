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
- AMulti-GCLSTM

UCTB is a flexible and open package. You can use the data we provided or use your own data, the data structure is well stated in the tutorial chapter. You can build your own model based on model-units we provided and use the model-training class to train the model.

## Installation

##### Step 1 : Install TensorFlow

You can skip to step 2 if you already installed tensorflow.

You can refer to this page <https://www.tensorflow.org/install> to install tensorflow, if you have a Nvidia GPU installed on you computer, we highly recommend you to install GPU version of tensorflow.

##### Step 2 : Install UCTB

```bash
pip install --upgrade UCTB
```

The following required package will be installed or upgraded with UCTB:

```bash
'hmmlearn>=0.2.1',
'numpy>=1.16.2',
'pandas>=0.24.2',
'python-dateutil',
'scikit-learn>=0.20.3',
'scipy>=1.2.1',
'statsmodels>=0.9.0',
'wget>=3.2',
'xgboost>=0.82'
```

## Quick start

- ##### Quick start with AMulti-GCLSTM

```python
from UCTB.dataset import NodeTrafficLoader
from UCTB.model import AMulti_GCLSTM
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC')

AMulti_GCLSTM_Obj = AMulti_GCLSTM(T=6, num_node=data_loader.station_number,
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim)

AMulti_GCLSTM_Obj.build()

# Training
AMulti_GCLSTM_Obj.fit(input=data_loader.train_x,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      external_feature=data_loader.train_ef)

# Evaluate
test_rmse = AMulti_GCLSTM_Obj.evaluate(input=data_loader.test_x,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       metrics=[metric.rmse],
                                       threshold=0)

print('Test result', test_rmse)
```

- [Quick start with other models](./static/quick_start.html)

## Tutorial

- ##### Use datasets from UCTB

- ##### [Build your own datasets](./static/tutorial.html)

- ##### Use build-in models from UCTB


- ##### Build your own model using UCTB


## API Reference

- ##### Model Class


- ##### ModelUnit Class


- ##### Evaluation Class


- ##### Training Class


## Full examples

- ##### Experiments on bike traffic-flow prediction

  - Experiment Setting
  - Experiment Results
  - Source Code
- ##### Experiments on subway traffic-flow prediction
- ##### Experiments on charge-station demand prediction

## Contribute to UCTB

- ##### Contribute Data
- ##### Contribute Models
- ##### Contribute Model-Units
