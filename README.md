# UCTB (Urban Computing Tool Box)

 [![Python](https://img.shields.io/badge/python-3.6%7C3.7-blue)]() [![PyPI](https://img.shields.io/badge/pypi%20package-v0.3.0-sucess)](https://pypi.org/project/UCTB/) ![tensorflow](https://img.shields.io/badge/tensorflow-1.13-important)[![https://img.shields.io/badge/license-MIT-green](https://img.shields.io/badge/license-MIT-green)]() 

------

### News

**2021-11**: Our paper on UCTB, entitled '*Exploring the Generalizability of Spatio-Temporal Traffic Prediction: Meta-Modeling and an Analytic Framework*', has been accepted by IEEE TKDE! [[IEEE Xplore](https://ieeexplore.ieee.org/document/9627543)][[arXiv](https://arxiv.org/abs/2009.09379)]

------

**Urban Computing Tool Box** is a package providing both **spatial-temporal prediction models** and **urban datasets** for various urban computing tasks, such as traffic prediction, crowd flow prediction, ridesharing demand prediction, etc. It contains both conventional models and state-of-art models. 

------

Currently the package supported the following models: (This tool box is constructed based on some open-source repos. We appreciate these awesome implements.  [See more details](https://uctb.github.io/UCTB/md_file/static/current_supported_models.html))

|  Model Name  |   Input Data Format   |   Spatial Modeling Technique   |Graph Type|Temporal Modeling Technique|Temporal Knowledge|Module|
| ---- | ---- | ---- |----|----|----|----|
|   ARIMA   |   Both   |   N/A   |N/A|SARIMA|Closeness|``UCTB.model.ARIMA``|
|   HM   |   Both   |   N/A   |N/A|N/A|Closeness|``UCTB.model.HM``|
|   HMM   |   Both   |   N/A   |N/A|HMM|Closeness|``UCTB.model.HMM``|
|   XGBoost   |   Both   |   N/A   |N/A|XGBoost|Closeness|``UCTB.model.XGBoost``|
|   DeepST [[SIGSPATIAL 2016]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/DeepST-SIGSPATIAL2016.pdf)  |   Grid   |   CNN   |N/A|CNN|Closeness,Period,Trend|``UCTB.model.DeepST``|
|   ST-ResNet [[AAAI 2017]](https://arxiv.org/pdf/1610.00081.pdf)  |   Grid   |   CNN   |N/A|CNN|Closeness,Period,Trend|``UCTB.model.ST_ResNet``|
|   DCRNN  [[ICLR 2018]](https://arxiv.org/pdf/1707.01926.pdf) |   Node   |   GNN   |**Prior**(Sensor Network)|RNN|Closeness|``UCTB.model.DCRNN``|
|   GeoMAN  [[IJCAI 2018]](https://www.ijcai.org/proceedings/2018/0476.pdf) |   Node   |   Attention   |**Prior**(Sensor Networks)|Attention+LSTM|Closeness|``UCTB.model.GeoMAN``|
|   STGCN  [[IJCAI 2018]](https://www.ijcai.org/proceedings/2018/0505.pdf) |   Node   |   GNN   |**Prior**(Traffic Network)|Gated CNN|Closeness|``UCTB.model.STGCN``|
|   GraphWaveNet [[IJCAI 2019]](https://www.ijcai.org/proceedings/2019/0264.pdf)  |   Node   |   GNN   |**Adaptive**|TCN|Closeness|``UCTB.model.GraphWaveNet``|
|   ASTGCN  [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/3881) |   Node   |   GNN+Attention   |**Prior**(Traffic Network)|Attention|Closeness,Period,Trend|``UCTB.model.ASTGCN``|
|   ST-MGCN   [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/4247) |   Node   |   GNN   |**Prior**(Neighborhood,Functional similarity,Transportation connectivity)|CGRNN|Closeness|``UCTB.model.ST_MGCN``|
|   GMAN  [[AAAI 2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5477/5333) |   Node   |   Attention   |**Prior**(Road Network)|Attention|Closeness|``UCTB.model.GMAN``|
|   STSGCN  [[AAAI 2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |   Node   |   GNN+Attention   |**Prior**(Spatial Network)|Attention|Closeness|``UCTB.model.STSGCN``|
|   AGCRN [[NeurIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf) |   Node   |   GNN   |**Adaptive**|RNN|Closeness|``UCTB.model.AGCRN``|
|   STMeta [[TKDE 2021]](https://arxiv.org/abs/2009.09379)  |   Node   |   GNN   |**Prior**(Proximity,Functionality,Interaction/Same-line)|LSTM/RNN|Closeness,Period,Trend|``UCTB.model.STMeta``|


------

UCTB also releases [a public dataset repository](https://github.com/uctb/Urban-Dataset) including the following applications:

- Bike sharing in NYC, Chicago and DC
- Ride sharing in Chicago
- Traffic speed in LA and BAY
- Pedestrian counting in Melbourne

We provide [detailed documents](https://github.com/uctb/Urban-Dataset/blob/main/Tutorial/tutorial.ipynb) about how to build and how to use these datasets.

------

UCTB is a flexible and open package. You can use the data we provided or use your own data, the data structure is well stated in the tutorial chapter. You can build your own model based on model-units we provided and use the model-training class to train the model.

UCTB toolbox may not work successfully with the upgrade of some packages. We thus encourage you to use the specific version of packages or use our docker environment to avoid these unseen errors.

```
python==3.6
tensorflow==1.13
Keras==2.2.4
h5py==2.9.0
```

[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://uctb.github.io/UCTB)
