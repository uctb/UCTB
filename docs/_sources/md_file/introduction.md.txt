## Introduction

**Urban Computing ToolBox** is a package providing spatial-temporal predicting models. It contains both conventional models and state-of-art models. 

Currently the package supported the following models: (This tool box is constructed based on some open-source repos. We appreciate these awesome implements.  [See more details](https://uctb.github.io/UCTB/md_file/static/current_supported_models.html))

- ARIMA
- HM
- HMM
- XGBoost
- DeepST [[SIGSPATIAL 2016]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/DeepST-SIGSPATIAL2016.pdf)
- ST-ResNet [[AAAI 2017]](https://arxiv.org/pdf/1610.00081.pdf)
- DCRNN [[ICLR 2018]](https://arxiv.org/pdf/1707.01926.pdf)
- GeoMAN [[IJCAI 2018]](https://www.ijcai.org/proceedings/2018/0476.pdf)
- STGCN [[IJCAI 2018]](https://www.ijcai.org/proceedings/2018/0505.pdf)
- GraphWaveNet [[IJCAI 2019]](https://www.ijcai.org/proceedings/2019/0264.pdf)
- ASTGCN [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/3881)
- ST-MGCN [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/4247)
- GMAN [[AAAI 2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5477/5333)
- STSGCN [[AAAI 2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5438)
- AGCRN [[NeurIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)
- STMeta [[TKDE 2021]](https://arxiv.org/abs/2009.09379)

------

UCTB also releases [a public dataset repository](https://github.com/uctb/Urban-Dataset) including the following applications:

- Bike sharing in NYC, Chicago and DC
- Ride sharing in Chicago
- Traffic speed in LA and BAY
- Pedestrian counting in Melbourne

We provide [detailed documents](https://github.com/uctb/Urban-Dataset/blob/main/Tutorial/tutorial.ipynb) about how to build and how to use these datasets.

UCTB is a flexible and open package. You can use the data we provided or use your own data, the data structure is well stated in the tutorial chapter. You can build your own model based on model-units we provided and use the model-training class to train the model.

You can view UCTB's source code at [UCTB ToolBox](https://github.com/uctb/UCTB).

<u>[Back To HomePage](../index.html)</u>

