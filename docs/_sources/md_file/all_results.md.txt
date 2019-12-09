# Results on different datasets

## STMeta Version

As introduced in [Currently Supported Models](./static/current_supported_models.html#stmeta), STMeta is a meta-model that can be implemented by different deep learning techniques based on its applications. Here we realize three versions of STMeta to evaluate its generalizability. The main differences between these three variants are the techniques used in spatio-temporal modeling and aggregation units:

| Version Name |                     Spatio-Temporal Unit                     |                  Temporal Aggregation Unit                   | Spatial Aggregation Unit |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------: |
|  STMeta-V1   | [GCLSTM](../UCTB.model_unit.html?highlight=gclstmcel#UCTB.model_unit.ST_RNN.GCLSTMCell) | [GAL](../UCTB.model_unit.html?highlight=gclstmcel#UCTB.model_unit.GraphModelLayers.GAL) |           GAL            |
|  STMeta-V2   |                            GCLSTM                            |                        Concat & Dense                        |           GAL            |
|  STMeta-V3   | [DCGRU](../UCTB.model_unit.html?highlight=gclstmcel#UCTB.model_unit.DCRNN_CELL.DCGRUCell) |                             GAL                              |           GAL            |

By default, we use `STMeta-V1` to run LSTM and single graph model tests.

References:

- GCLSTM (Graph Convolutional Long short-term Memory):
  [Chai, D., Wang, L., & Yang, Q. (2018, November). Bike flow prediction with multi-graph convolutional networks](https://arxiv.org/pdf/1807.10934) 

- DCGRU (Diffusion Convolutional Gated Recurrent Unit):
  [Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting](https://arxiv.org/pdf/1707.01926.pdf)  

- GAL (Graph Attention Layer):
  [Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks](https://arxiv.org/pdf/1710.10903.pdf) 

Our running code and detailed parameter settings can be found in [Experiment Setting](./all_results_setting.html).

## Results on Bike

### Dataset Statistics

|        Attributes        | **New York City** |   **Chicago**   |     **DC**      |
| :----------------------: | :---------------: | :-------------: | :-------------: |
|        Time span         |  2013.03-2017.09  | 2013.07-2017.09 | 2013.07-2017.09 |
| Number of riding records |    49,100,694     |   13,130,969    |   13,763,675    |
|    Number of stations    |        820        |       585       |       532       |

 Following shows a map-visualization of bike stations in NYC.

![NYC](../src/image/NYC.jpg)

### Experiment Results

|                 |   **NYC**   | **Chicago** |   **DC**    |
| :-------------: | :---------: | :---------: | :---------: |
|       HM        |   3.99224   |   2.97693   |   2.63165   |
|    ARIMA(C)     |   5.60928   |   3.83584   |   3.60450   |
|     XGBoost     |   4.12407   |   2.92569   |   2.65671   |
|      GBRT       |   3.99907   |   2.84257   |   2.61768   |
| ST_MGCN (G/DCI) |   3.72380   |   2.88300   |   2.48560   |
|  DCRNN(G/D C)   |   4.18666   |   3.27759   |   3.08616   |
|    LSTM (C)     |   4.55677   |   3.37004   |   2.91518   |
|    STMeta-V1    |   3.50475   | **2.65511** |   2.42582   |
|    STMeta-V2    | **3.43870** |   2.66379   |   2.41111   |
|    STMeta-V3    |   3.47834   |   2.66180   | **2.38844** |



## Results on DiDi

### Dataset Statistics

|        Attributes        |    **Xi'an**    |   **Chengdu**   |
| :----------------------: | :-------------: | :-------------: |
|        Time span         | 2016.10-2016.11 | 2016.10-2016.11 |
| Number of riding records |    5,922,961    |    8,439,537    |
|    Number of stations    |       256       |       256       |

Following shows a map-visualization of 256 grid-based ride-sharing stations in Chengdu.

![](../src/image/Chengdu.jpg)

### Experiment Results

|                   |  **Xi’an**  | **Chengdu**  |
| :---------------: | :---------: | :----------: |
|        HM         |   6.18623   |   7.35477    |
|     ARIMA(C)      |   9.47478   |   12.52656   |
|      XGBoost      |   6.73346   |   7.73836    |
|       GBRT        |   6.44639   |   7.58831    |
|     ST-ResNet     |   6.08476   |   7.14638    |
|  ST_MGCN (G/DCI)  |   5.87456   | **7.03217 ** |
|   DCRNN(G/D C)    |   8.20254   |   11.50550   |
|     LSTM (C)      |   7.39970   |   10.11386   |
| STMeta-V1 (G/DCI) |   5.89154   |   7.06246    |
| STMeta-V2(G/DCI)  | **5.75596** |   7.09710    |
| STMeta-V3(G/DCI)  |   5.95507   |   7.04358    |

## Results on Metro

### Dataset Statistics

|        Attributes        |  **Chongqing**  |  **Shanghai**   |
| :----------------------: | :-------------: | :-------------: |
|        Time span         | 2016.08-2017.07 | 2016.07-2016.09 |
| Number of riding records |   409,277,117   |   333,149,034   |
|    Number of stations    |       113       |       288       |

Following shows a map-visualization of 288 metro stations in Shanghai.

![](../src/image/Shanghai.jpg)

### Experiment Results

|                   | **Chongqing** | **Shanghai**  |
| :---------------: | :-----------: | :-----------: |
|        HM         |   120.30723   |   197.97092   |
|     ARIMA(C)      |   578.18563   |   792.1597    |
|      XGBoost      |   117.05069   |   185.00447   |
|       GBRT        |   113.92276   |   186.74502   |
|  ST_MGCN (G/DCI)  |   118.86668   |   181.55171   |
|   DCRNN(G/D C)    |   122.31121   |   326.97357   |
|     LSTM (C)      |  196.175732   |   368.8468    |
| STMeta-V1 (G/DCI) | **92.74990**  | **151.11746** |
| STMeta-V2(G/DCI)  |   98.86152    |   158.21953   |
| STMeta-V3(G/DCI)  |   101.7806    |   156.58867   |

The period and trend features are more obvious in Metro dataset, so the performance is poor if only use closeness feature.

## Results on Charge-Station

### Dataset Statistics

|        Attributes        |   **Beijing**   |
| :----------------------: | :-------------: |
|        Time span         | 2018.03-2018.08 |
| Number of riding records |    1,272,961    |
|    Number of stations    |       629       |

Following shows a map-visualization of  629 EV charging stations in Beijing.

![](../src/image/Beijing.jpg)

### Experiment Results

|                  | **Beijing**  |
| :--------------: | :----------: |
|        HM        |   1.01610    |
|     ARIMA(C)     |   0.98236    |
|     XGBoost      |   0.83381    |
|       GBRT       |   0.82814    |
|  ST_MGCN (G/DC)  |   0.82714    |
|   DCRNN(G/D C)   |   0.98871    |
|     LSTM (C)     |   1.58560    |
| STMeta-V1 (G/DC) | **0.815518** |
| STMeta-V2(G/DC)  |   0.82144    |
| STMeta-V3(G/DC)  | **0.81541**  |

