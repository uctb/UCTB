# UCTB (Urban Computing Tool Box)

 [![Python](https://img.shields.io/badge/python-3.6%7C3.7-blue)]() [![PyPI](https://img.shields.io/badge/pypi%20package-v0.3.5-sucess)](https://pypi.org/project/UCTB/) [![https://img.shields.io/badge/license-MIT-green](https://img.shields.io/badge/license-MIT-green)]() [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://uctb.github.io/UCTB)

------

**Urban Computing Tool Box** is a package providing [**ST paper list**](https://github.com/uctb/ST-Paper), [**urban datasets**](https://github.com/uctb/Urban-Dataset), [**spatial-temporal prediction models**](https://github.com/uctb/UCTB), and [**visualization tools**](https://github.com/uctb/visualization-tool-UCTB) for various urban computing tasks, such as traffic prediction, crowd flow prediction, ride-sharing demand prediction, etc. 

UCTB is a flexible and open package. You can use the data we provided or use your data, and the data structure is well stated in the [**document**](https://uctb.github.io/UCTB/md_file/urban_dataset.html#). 

## News

**2024-03**: We have released two new datasets for **Metro** and **Bus** applications. These datasets provide hourly estimates of subway and bus ridership. [Welcome to explore them!](https://github.com/uctb/Urban-Dataset)

**2023-06**: We have released a technical report entitled '*UCTB: An Urban Computing Tool Box for Spatiotemporal Crowd Flow Prediction*', introducing the design and implementation principles of UCTB. [[arXiv\]](https://arxiv.org/abs/2306.04144)

**2021-11**: Our paper on UCTB, entitled '*Exploring the Generalizability of Spatio-Temporal Traffic Prediction: Meta-Modeling and an Analytic Framework*', has been accepted by IEEE TKDE! [[IEEE Xplore](https://ieeexplore.ieee.org/document/9627543)] [[arXiv](https://arxiv.org/abs/2009.09379)]

------

## ST-Paper List

We maintain [a paper list]((https://github.com/uctb/ST-Paper)) focusing on spatio-temporal prediction papers from venues such as KDD, NeurIPS, AAAI, WWW, ICDE, IJCAI, WSDM, CIKM, and IEEE T-ITS. Note that the metadata may not encompass all relevant papers and could include unrelated ones, as selected by large language models.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/venue_stat.png" alt=".img" style="zoom: 33%;height: 327px; width:424" />

## Urban Datasets

UCTB releases [a public dataset repository](https://github.com/uctb/Urban-Dataset) including the following applications in 4 scenarios, with the detailed information provided in the table below. We are constantly working to release more datasets in the future.

| **Application**  |        **City**        |       Time Span       | **Interval** |                           **Link**                           |
| :--------------: | :--------------------: | :-------------------: | :----------: | :----------------------------------------------------------: |
|   Bike-sharing   |          NYC           | 2013.07.01-2017.09.30 | 5 & 60 mins  | [5 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/5_minutes/Bike_NYC.zip)  [60 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/60_minutes/Bike_NYC.zip) |
|   Bike-sharing   |        Chicago         | 2013.07.01-2017.09.30 | 5 & 60 mins  | [5 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/5_minutes/Bike_Chicago.zip) [60 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/60_minutes/Bike_Chicago.zip) |
|   Bike-sharing   |           DC           | 2013.07.01-2017.09.30 | 5 & 60 mins  | [5 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/5_minutes/Bike_DC.zip) [60 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bike/60_minutes/Bike_DC.zip) |
|       Bus        |          NYC           | 2022.02.01-2024.01.13 |   60 mins    | [60 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Bus/60_minutes/Bus_NYC.zip) |
|  Vehicle Speed   |           LA           | 2012.03.01-2012.06.28 |    5 mins    | [5 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Speed/5_minutes/METR_LA.zip) |
|  Vehicle Speed   |          BAY           | 2017.01.01-2017.07.01 |    5 mins    | [5 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Speed/5_minutes/PEMS_BAY.zip) |
| Pedestrian Count |       Melbourne        | 2021.01.01-2022.11.01 |   60 mins    | [60 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Pedestrian/60_minutes/Pedestrian_Melbourne.zip) |
|   Ride-sharing   |  Chicago (community)   | 2013.01.01-2018.01.01 |   15 mins    | [15 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Taxi/15_minutes/Taxi_Chicago.zip) |
|   Ride-sharing   | Chicago (census tract) | 2013.01.01-2018.01.01 |   15 mins    | [15 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Taxi/15_minutes/Taxi_fine_grained_Chicago.zip) |
|   Ride-sharing   |          NYC           | 2009.01.01-2023.06.01 |    5 mins    | [5 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Taxi/5_minutes/Taxi_NYC.zip) |
|      Metro       |          NYC           | 2022.02.01-2023.12.21 |   60 mins    | [60 mins](https://github.com/uctb/Urban-Dataset/blob/main/Public_Datasets/Metro/60_minutes/Metro_NYC.zip) |

We provide [detailed documents](https://github.com/uctb/Urban-Dataset/blob/main/Tutorial/tutorial.ipynb) about how to use these datasets.

------

## Prediction Models

Currently, the ST prediction model package supports the following models: (This toolbox is constructed based on some open-source repos. We appreciate these awesome implements. [See more details](https://uctb.github.io/UCTB/md_file/predictive_tool.html#)). 

|  Model  |   Data Format   |   Spatial Modeling Technique   |Graph Type|Temporal Modeling Technique|Temporal Knowledge|Module|
| :--: | :--: | :--: |:--:|:--:|:--:|:--:|
|   ARIMA   |   Both   |   N/A   |N/A|SARIMA|Closeness|``UCTB.model.ARIMA``|
|   HM   |   Both   |   N/A   |N/A|N/A|Closeness|``UCTB.model.HM``|
|   HMM   |   Both   |   N/A   |N/A|HMM|Closeness|``UCTB.model.HMM``|
|   XGBoost   |   Both   |   N/A   |N/A|XGBoost|Closeness|``UCTB.model.XGBoost``|
|   DeepST [[SIGSPATIAL 2016]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/DeepST-SIGSPATIAL2016.pdf)  |   Grid   |   CNN   |N/A|CNN|Closeness, Period, Trend|``UCTB.model.DeepST``|
|   ST-ResNet [[AAAI 2017]](https://arxiv.org/pdf/1610.00081.pdf)  |   Grid   |   CNN   |N/A|CNN|Closeness, Period, Trend|``UCTB.model.ST_ResNet``|
|   DCRNN [[ICLR 2018]](https://arxiv.org/pdf/1707.01926.pdf) |   Node   |   GNN   |**Prior** (Sensor Network)|RNN|Closeness|``UCTB.model.DCRNN``|
|   GeoMAN [[IJCAI 2018]](https://www.ijcai.org/proceedings/2018/0476.pdf) |   Node   |   Attention   |**Prior** (Sensor Networks)|Attention+LSTM|Closeness|``UCTB.model.GeoMAN``|
|   STGCN [[IJCAI 2018]](https://www.ijcai.org/proceedings/2018/0505.pdf) |   Node   |   GNN   |**Prior** (Traffic Network)|Gated CNN|Closeness|``UCTB.model.STGCN``|
|   GraphWaveNet [[IJCAI 2019]](https://www.ijcai.org/proceedings/2019/0264.pdf)  |   Node   |   GNN   |**Prior** (Sensor Network) + **Adaptive**|TCN|Closeness|``UCTB.model.GraphWaveNet``|
|   ASTGCN [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/3881) |   Node   |   GNN+Attention   |**Prior** (Traffic Network)|Attention|Closeness, Period, Trend|``UCTB.model.ASTGCN``|
|  ST-MGCN [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/4247) |   Node   |   GNN   |**Prior** (Neighborhood, Functional similarity, Transportation connectivity)|CGRNN|Closeness|``UCTB.model.ST_MGCN``|
|   GMAN [[AAAI 2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5477/5333) |   Node   |   Attention   |**Prior** (Road Network)|Attention|Closeness|``UCTB.model.GMAN``|
|   STSGCN [[AAAI 2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |   Node   |   GNN+Attention   |**Prior** (Spatial Network)|Attention|Closeness|``UCTB.model.STSGCN``|
|  AGCRN [[NeurIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf) |   Node   |   GNN   |**Adaptive**|RNN|Closeness|``UCTB.model.AGCRN``|
|  AGCRN [[KDD 2020]](https://dl.acm.org/doi/abs/10.1145/3394486.3403118) |   Node   |   GNN   |**Adaptive**|TCN|Closeness|``UCTB.model.MTGNN``|
|   STMeta [[TKDE 2021]](https://arxiv.org/abs/2009.09379)  |   Node   |   GNN   |**Prior** (Proximity, Functionality, Interaction/Same-line)|LSTM/RNN|Closeness, Period, Trend|``UCTB.model.STMeta``|

------

## Visualization Tool

The Visualization tool integrates visualization, error localization, and error diagnosis. Specifically, it allows data to be uploaded and provides interactive visual charts to show model errors, combined with spatiotemporal knowledge for error diagnosis.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/vis_5.png" alt=".img" style="zoom: 33%;" />

Welcome to visit the [website](http://39.107.116.221/) for a trial! 

## Installation

UCTB toolbox may not work successfully with the upgrade of some packages. We thus encourage you to use the specific version of packages to avoid unseen errors. ***To avoid potential conflict, we highly recommend you install UCTB vis Anaconda.*** The installation details are in our [documents](https://uctb.github.io/UCTB/md_file/installation.html). 

## Citation

If UCTB is helpful for your work, please cite and star our project.

```
@article{uctb_2023,
  title={UCTB: An Urban Computing Tool Box for Spatiotemporal Crowd Flow Prediction},
  author={Chen, Liyue and Chai, Di and Wang, Leye},
  journal={arXiv preprint arXiv:2306.04144},
  year={2023}}

@article{STMeta,
  author={Wang, Leye and Chai, Di and Liu, Xuanzhe and Chen, Liyue and Chen, Kai},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Exploring the Generalizability of Spatio-Temporal Traffic Prediction: Meta-Modeling and an Analytic Framework}, 
  year={2023},
  volume={35},
  number={4},
  pages={3870-3884},
  doi={10.1109/TKDE.2021.3130762}}
```
