#   Results on different datasets

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

We conducted experiments on the following datasets at the granularity of 15 minutes, 30 minutes and 60 minutes respectively. Our running code and detailed parameter settings can be found in [Experiment Setting](./all_results.html#experiment-setting-on-different-datasets).

## Results on Bike

### Dataset Statistics

|        Attributes        | **New York City** |   **Chicago**   |     **DC**      |
| :----------------------: | :---------------: | :-------------: | :-------------: |
|        Time span         |  2013.03-2017.09  | 2013.07-2017.09 | 2013.07-2017.09 |
| Number of riding records |    49,100,694     |   13,130,969    |   13,763,675    |
|    Number of stations    |        820        |       585       |       532       |

Following shows the map-visualization of bike stations in NYC, Chicago and DC.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Bike_NYC.jpg" style="zoom:30%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Bike_Chicago.jpg" style="zoom:30%;height:800px;width:800px;"/> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Bike_DC.jpg" style="zoom:30%;height:800px;width:800px;" />

### Experiment Results

| **15 minutes** |     NYC     |   Chicago   |     DC      |
| :------------: | :---------: | :---------: | :---------: |
|       HM       |   1.89180   |   1.66782   |   1.55471   |
|     ARIMA      |   1.87415   |   1.78399   |   1.68858   |
|    XGBoost     |   1.71216   |   1.67219   |   1.55872   |
|      GBRT      |   1.70757   |   1.66691   |   1.55246   |
|    ST_MGCN     |   1.68659   |   1.64642   |   1.54455   |
|     DCRNN      |   1.71223   |   1.71789   |   1.59412   |
|      LSTM      |   1.98866   |   1.80222   |   1.67762   |
| TMeta-LSTM-GAL |   1.81819   |   1.62269   |   1.54041   |
|   STMeta-V1    |   1.65939   | **1.60743** |   1.52698   |
|   STMeta-V2    |   1.67336   |   1.62883   | **1.51158** |
|   STMeta-V3    | **1.65351** |   1.60917   |   1.51720   |

| **30 minutes** |     NYC     |   Chicago   |     DC      |
| :------------: | :---------: | :---------: | :---------: |
|       HM       |   2.68564   |   2.22987   |   1.95601   |
|     ARIMA      |   3.17849   |   2.42798   |   2.22804   |
|    XGBoost     |   2.70377   |   2.37553   |   1.95560   |
|      GBRT      |   2.68164   |   2.35532   |   1.92799   |
|    ST_MGCN     |   2.51288   |   2.17659   |   1.90305   |
|     DCRNN      |   2.61848   |   2.24642   |   2.11771   |
|      LSTM      |   3.01836   |   2.49270   |   2.21191   |
| TMeta-LSTM-GAL |   2.51124   |   2.13333   |   1.92748   |
|   STMeta-V1    | **2.40976** |   2.17032   |   1.85628   |
|   STMeta-V2    |   2.41088   | **2.13330** |   1.85876   |
|   STMeta-V3    |   2.41109   |   2.18174   | **1.85199** |

| **60 minutes** |    NYC    |  Chicago  |    DC     |
| :------------: | :-------: | :-------: | :-------: |
|       HM       |   3.992   |   2.976   |   2.631   |
|     ARIMA      |   5.609   |   3.835   |   3.604   |
|    XGBoost     |   4.124   |   2.925   |   2.656   |
|      GBRT      |   3.999   |   2.842   |   2.617   |
|    ST_MGCN     |   3.723   |   2.883   |   2.485   |
|     DCRNN      |   4.186   |   3.277   |   3.086   |
|      LSTM      |   4.556   |   3.370   |   2.915   |
| TMeta-LSTM-GAL |   3.784   |   2.790   |   2.547   |
|   STMeta-V1    |   3.504   | **2.655** |   2.425   |
|   STMeta-V2    | **3.438** |   2.663   |   2.411   |
|   STMeta-V3    |   3.478   |   2.661   | **2.388** |

## Results on Metro

### Dataset Statistics

|     Attributes     |  **Chongqing**  |  **Shanghai**   |
| :----------------: | :-------------: | :-------------: |
|     Time span      | 2016.08-2017.07 | 2016.07-2016.09 |
| Number of records  |   409,277,117   |   333,149,034   |
| Number of stations |       113       |       288       |

Following shows the map-visualization of metro stations in Chongqing and Shanghai.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Metro_Chongqing.jpg" style="zoom:30%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Metro_Shanghai.jpg" style="zoom:30%;height:800px;width:800px;" />

### Experiment Results

|   15 minutes   | **Chongqing** | **Shanghai** |
| :------------: | :-----------: | :----------: |
|       HM       |   45.25524    |   49.74561   |
|     ARIMA      |   67.11072    |   83.53750   |
|    XGBoost     |   35.69683    |   47.88690   |
|      GBRT      |   33.28726    |   44.55068   |
|    ST_MGCN     |   32.71874    |   46.54292   |
|     DCRNN      |   37.06903    |   56.00411   |
|      LSTM      |   55.36633    |   80.40264   |
| TMeta-LSTM-GAL |   33.34361    |   45.88331   |
|   STMeta-V1    | **31.39239**  |   41.66834   |
|   STMeta-V2    |   38.20912    |   43.82808   |
|   STMeta-V3    |   36.90250    | **40.94003** |

| **30 minutes** |  Chongqing   |   Shanghai   |
| :------------: | :----------: | :----------: |
|       HM       |   74.54662   |  108.59372   |
|     ARIMA      |  180.53262   |  212.00777   |
|    XGBoost     |   69.50227   |   81.82434   |
|      GBRT      |   72.98518   |   83.93989   |
|    ST_MGCN     |   50.95764   |   88.76412   |
|     DCRNN      |   65.71969   |  116.14510   |
|      LSTM      |  104.60832   |  195.60097   |
| TMeta-LSTM-GAL |   53.17723   |   85.19422   |
|   STMeta-V1    |   49.46800   | **75.36282** |
|   STMeta-V2    |   50.01080   |   80.68939   |
|   STMeta-V3    | **48.95798** |   77.48744   |

|   60 minutes   | **Chongqing** | **Shanghai** |
| :------------: | :-----------: | :----------: |
|       HM       |    120.30     |    197.97    |
|     ARIMA      |    578.18     |    792.15    |
|    XGBoost     |    117.05     |    185.00    |
|      GBRT      |    113.92     |    186.74    |
|    ST_MGCN     |    118.86     |    181.55    |
|     DCRNN      |    122.31     |    326.97    |
|      LSTM      |    196.17     |    368.84    |
| TMeta-LSTM-GAL |     97.50     |    182.28    |
|   STMeta-V1    |   **92.74**   |  **151.11**  |
|   STMeta-V2    |     98.86     |    158.21    |
|   STMeta-V3    |    101.78     |    156.58    |

## Results on Charge-Station

### Dataset Statistics

|     Attributes     |   **Beijing**   |
| :----------------: | :-------------: |
|     Time span      | 2018.03-2018.05 |
| Number of records  |    1,272,961    |
| Number of stations |       629       |

Following shows the map-visualization of  629 EV charging stations in Beijing.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/EV_Beijing.jpg" style="zoom:40%;height:800px;width:800px;" />

### Experiment Results

|   30 minutes   | **Beijing** |
| :------------: | :---------: |
|       HM       |   0.86361   |
|     ARIMA      |   0.75522   |
|    XGBoost     |   0.68649   |
|      GBRT      |   0.68931   |
|    ST_MGCN     |   0.69083   |
|     DCRNN      |   0.75740   |
|      LSTM      |   0.75474   |
| TMeta-LSTM-GAL |   0.68627   |
|   STMeta-V1    |   0.66985   |
|   STMeta-V2    | **0.66675** |
|   STMeta-V3    |   0.66966   |

|   60 minutes   | **Beijing** |
| :------------: | :---------: |
|       HM       |    1.016    |
|     ARIMA      |    0.982    |
|    XGBoost     |    0.833    |
|      GBRT      |    0.828    |
|    ST_MGCN     |    0.827    |
|     DCRNN      |    0.988    |
|      LSTM      |    1.585    |
| TMeta-LSTM-GAL |    0.833    |
|   STMeta-V1    |  **0.815**  |
|   STMeta-V2    |    0.821    |
|   STMeta-V3    |  **0.815**  |

## Results on Traffic Speed

### Dataset Statistics

|        Attributes        |   **METR-LA**   |  **PEMS-BAY**   |
| :----------------------: | :-------------: | :-------------: |
|        Time span         | 2012.03-2012.06 | 2017.01-2017.07 |
| Number of riding records |     34,272      |     52,128      |
|    Number of stations    |       207       |       325       |

Following shows the map-visualization of grid-based ride-sharing stations in METR-LA and PEMS-BAY.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/METR_LA.png" style="zoom:30%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/PEMS_BAY.png" style="zoom:30%;height:800px;width:800px;" /> 

### Experiment Results

|   15 minutes   | **METR-LA** | **PEMS-BAY** |
| :------------: | :---------: | :----------: |
|       HM       |   8.93415   |   3.68983    |
|     ARIMA      |   7.02787   |   2.86893    |
|    XGBoost     |   6.44322   |   2.62339    |
|      GBRT      |   6.37050   |   2.64524    |
|    ST_MGCN     |   6.64489   |   2.42605    |
|     DCRNN      |   6.44030   |   5.32297    |
|      LSTM      |   6.38015   |   2.68953    |
| TMeta-LSTM-GAL |   6.15585   |   2.54368    |
|   STMeta-V1    |   5.64445   |   2.43292    |
|   STMeta-V2    |   5.79998   |   2.44947    |
|   STMeta-V3    |   5.78807   |   2.44571    |

|   30 minutes   | **METR-LA** | **PEMS-BAY** |
| :------------: | :---------: | :----------: |
|       HM       |   9.55981   |   3.96537    |
|     ARIMA      |   9.22951   |   3.93569    |
|    XGBoost     |   8.29796   |   3.25334    |
|      GBRT      |   8.26941   |   3.37025    |
|    ST_MGCN     |   8.07924   |   3.04172    |
|     DCRNN      |   8.56215   |   6.19802    |
|      LSTM      |   7.86569   |   3.68256    |
| TMeta-LSTM-GAL |   7.43553   |   3.23098    |
|   STMeta-V1    |   7.15628   |   3.11554    |
|   STMeta-V2    |   6.88889   |   3.20407    |
|   STMeta-V3    |   7.18431   |   3.18722    |

|   60 minutes   | **METR-LA** | **PEMS-BAY** |
| :------------: | :---------: | :----------: |
|       HM       |  10.72724   |   4.01788    |
|     ARIMA      |  11.73901   |   5.67008    |
|    XGBoost     |  10.29861   |   3.70330    |
|      GBRT      |  10.01320   |   3.70401    |
|    ST_MGCN     |  10.79813   |   3.48569    |
|     DCRNN      |  11.12053   |   6.91955    |
|      LSTM      |  10.08317   |   4.77696    |
| TMeta-LSTM-GAL |   8.66965   |   3.61642    |
|   STMeta-V1    |   8.83393   |   3.51389    |
|   STMeta-V2    |   9.14697   |   3.55159    |
|   STMeta-V3    |   8.99345   |   3.49954    |

## Experiment Setting on different datasets

### Search Space

We use [nni](https://github.com/microsoft/nni) toolkit to search the best parameters of HM, XGBoost and GBRT model. Search space are following.

|  Model  |                         Search Space                         |
| :-----: | :----------------------------------------------------------: |
|   HM    |               `CT: 0~6`, `PT: 0~7`, `TT: 0~4`                |
|  ARIMA  |                ``CT:168``,`p:3`, `d:0`, `q:0`                |
| XGBoost | `CT: 0~12`, `PT: 0~14`, `TT: 0~4`, `estimater: 10~200`, `depth: 2~10` |
|  GBRT   | `CT: 0~12`, `PT: 0~14`, `TT: 0~4`, `estimater: 10~200`, `depth: 2~10` |

### Results on Bike

#### Dataset Statistics

|        Attributes        | **New York City** |   **Chicago**   |     **DC**      |
| :----------------------: | :---------------: | :-------------: | :-------------: |
|        Time span         |  2013.03-2017.09  | 2013.07-2017.09 | 2013.07-2017.09 |
| Number of riding records |    49,100,694     |   13,130,969    |   13,763,675    |
|    Number of stations    |        820        |       585       |       532       |

#### Experiment Setting

* HM & XGBoost & GBRT

  | 15 minutes |      NYC       |    Chicago     |       DC       |
  | :--------: | :------------: | :------------: | :------------: |
  |     HM     |    `3-1-2`     |    `5-0-4`     |    `3-7-4`     |
  |  XGBoost   | `8-14-4-32-2`  | `11-13-4-28-2` | `4-14-4-27-2`  |
  |    GBRT    | `7-13-4-144-1` | `7-15-4-101-2` | `8-11-5-101-2` |

  | 30 minutes |      NYC       |    Chicago    |       DC        |
  | :--------: | :------------: | :-----------: | :-------------: |
  |     HM     |    `2-1-2`     |    `3-2-1`    |     `3-1-4`     |
  |  XGBoost   | `12-8-1-36-3`  | `7-5-2-24-2`  | `12-13-4-27-2`  |
  |    GBRT    | `12-10-0-72-4` | `9-13-2-91-2` | `13-15-5-140-1` |

  | 60 minutes |      NYC       |    Chicago    |      DC       |
  | :--------: | :------------: | :-----------: | :-----------: |
  |     HM     |    `1-1-3`     |    `1-1-1`    |    `2-1-3`    |
  |  XGBoost   | `13-7-0-103-3` | `11-8-0-35-4` | `11-9-5-28-3` |
  |    GBRT    | `12-6-1-58-5`  | `11-8-1-92-5` | `11-8-5-54-3` |

* [ST_MGCN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/bike_trial.py) Run Code & Setting.

* [DCRNN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/bike_trial.py) Run Code & Setting.

* LSTM & TMeta-LSTM-GAL & STMeta-V1  & STMeta-V2  & STMeta-V3

  These five models can be run by specifying data files and model files on  [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [bike_nyc.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/bike_nyc.data.yml) , [bike_chicago.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/bike_chicago.data.yml), [bike_dc.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/bike_dc.data.yml)

  Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml), [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).,  [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml)., [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

  * LSTM

  ```python
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
            ' -p data_range:0.25,train_data_length:91,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
            ' -p data_range:0.5,train_data_length:183,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
            ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
            ' -p data_range:0.25,train_data_length:91,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
            ' -p data_range:0.5,train_data_length:183,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
            ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
            ' -p data_range:0.25,train_data_length:91,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
            ' -p data_range:0.5,train_data_length:183,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
            ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
  ```

  * TMeta-LSTM-GAL

  ```python
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p data_range:0.25,train_data_length:91,graph:Distance,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p data_range:0.5,train_data_length:183,graph:Distance,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p graph:Distance,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml -p data_range:0.25,train_data_length:91,graph:Distance,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml -p data_range:0.5,train_data_length:183,graph:Distance,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml -p graph:Distance,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml -p data_range:0.25,train_data_length:91,graph:Distance,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml -p data_range:0.5,train_data_length:183,graph:Distance,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml -p graph:Distance,MergeIndex:12')
  ```

  * STMeta-V1

  ```python
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  ```

  * STMeta-V2

  ```python
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  ```

  * STMeta-V3

  ```python
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
            '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
            '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
  os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
            '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
  ```

The result of Bike dataset can be found [here](./all_results.html#experiment-results).

### Results on Metro

#### Dataset Statistics

|     Attributes     |  **Chongqing**  |  **Shanghai**   |
| :----------------: | :-------------: | :-------------: |
|     Time span      | 2016.08-2017.07 | 2016.07-2016.09 |
| Number of records  |   409,277,117   |   333,149,034   |
| Number of stations |       113       |       288       |

#### Experiment Setting

* HM & XGBoost & GBRT

  | 15 minutes |  **Chongqing**  |  **Shanghai**  |
  | :--------: | :-------------: | :------------: |
  |     HM     |     `2-1-4`     |    `1-0-4`     |
  |  XGBoost   |  `12-6-4-51-8`  | `11-10-4-31-7` |
  |    GBRT    | `12-14-1-182-7` | `12-7-1-148-5` |

  | 30 minutes | **Chongqing**  |  **Shanghai**  |
  | :--------: | :------------: | :------------: |
  |     HM     |    `1-0-4`     |    `1-1-3`     |
  |  XGBoost   | `11-5-0-45-8`  | `12-6-1-206-3` |
  |    GBRT    | `10-3-0-107-8` |  `7-4-1-58-7`  |

  | 60 minutes |  **Chongqing**   | **Shanghai** |
  | :--------: | :--------------: | :----------: |
  |     HM     |     `0-1-4`      |   `0-0-4`    |
  |  XGBoost   |  `9-14-2-200-5`  | `3-7-0-50-5` |
  |    GBRT    | `12-10-4-200-5 ` | `9-5-1-66-6` |

* [ST_MGCN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/metro_trial.py ) Run Code & Setting.

* [DCRNN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/metro_trial.py) Run Code & Setting.

* LSTM & TMeta-LSTM-GAL & STMeta-V1  & STMeta-V2  & STMeta-V3

  These five models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [metro_chongqing.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/metro_chongqing.data.yml) , [metro_shanghai.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/metro_shanghai.data.yml).

  Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml), [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).,  [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml)., [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

  * LSTM 
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
    ```
    
  * TMeta-LSTM-GAL
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml -p graph:Distance,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml -p graph:Distance,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml -p graph:Distance,MergeIndex:12')
    ```
  
  * STMeta-V1
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:12')
    ```
  
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:12')
    ```
    
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
            '-p graph:Distance-Correlation-Line,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
              '-p graph:Distance-Correlation-Line,MergeIndex:12')
    ```
    

The result of Metro dataset can be found [here](./all_results.html#id4).

### Results on Charge-Station

#### Dataset Statistics

|     Attributes     |   **Beijing**   |
| :----------------: | :-------------: |
|     Time span      | 2018.03-2018.05 |
| Number of records  |    1,272,961    |
| Number of stations |       629       |

#### Experiment Setting

* HM & XGBoost & GBRT

  | Beijing |  30 minutes   |   60 minutes    |
  | :-----: | :-----------: | :-------------: |
  |   HM    |    `2-0-0`    |     `2-0-2`     |
  | XGBoost | `6-6-1-19-2`  |  `12-7-0-20-2`  |
  |  GBRT   | `13-3-2-47-3` | `12-10-0-100-2` |

* [ST_MGCN]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/cs_trial.py ) Run Code & Setting.

* [DCRNN]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/cs_trial.py ) Run Code & Setting.

* LSTM & TMeta-LSTM-GAL & STMeta-V1  & STMeta-V2  & STMeta-V3

  These five models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [chargestation_beijing.data.yml]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/chargestation_beijing.data.yml).

  Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml), [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).,  [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml)., [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

  * LSTM 
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d chargestation_beijing.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:1')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d chargestation_beijing.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:2')
    ```
  
  * TMeta-LSTM-GAL
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:1')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:2')
    ```
  
  * STMeta-V1
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:1')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
            ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:2')
    ```
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:1')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
            ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:2')
    ```
  
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:1')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:2')
    ```

The result of Charge-Station dataset can be found [here](./all_results.html#id6).

### Results on Traffic Speed

#### Dataset Statistics

|        Attributes        |   **METR-LA**   |  **PEMS-BAY**   |
| :----------------------: | :-------------: | :-------------: |
|        Time span         | 2012.03-2012.06 | 2017.01-2017.07 |
| Number of riding records |     34,272      |     52,128      |
|    Number of stations    |       207       |       325       |

#### Experiment Setting

* HM & XGBoost & GBRT

  | 15 minutes |  **METR-LA**  | **PEMS-BAY**  |
  | :--------: | :-----------: | :-----------: |
  |     HM     |    `2-0-4`    |    `1-0-1`    |
  |  XGBoost   | `11-1-2-25-3` | `12-4-2-21-4` |
  |    GBRT    | `11-8-2-29-4` | `10-6-1-65-6` |

  | 30 minutes |  **METR-LA**  |  **PEMS-BAY**  |
  | :--------: | :-----------: | :------------: |
  |     HM     |    `2-0-4`    |    `1-0-1`     |
  |  XGBoost   | `6-6-0-25-3`  | `12-13-2-27-3` |
  |    GBRT    | `10-0-0-27-3` | `12-6-2-90-7`  |

  | 60 minutes |  **METR-LA**  | **PEMS-BAY**  |
  | :--------: | :-----------: | :-----------: |
  |     HM     |    `2-1-4`    |    `1-1-4`    |
  |  XGBoost   | `2-2-0-25-3`  | `12-6-2-19-3` |
  |    GBRT    | `4-5-1-19-4`  | `12-7-2-59-5` |

* [METR-LA](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/metr_trial.py)  and [PEMS-BAY](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/pems_trial.py)  ST_MGCN Run Code & Setting.

* [METR-LA](https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/metr_trial.py) and [PEMS-BAY](https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/pems_trial.py) DCRNN Run Code & Setting.

* LSTM & TMeta-LSTM-GAL & STMeta-V1  & STMeta-V2  & STMeta-V3

  These five models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [metr_la.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/metr_la.data.yml) , [pems_bay.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/pems_bay.data.yml).

  Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml), [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).,  [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml)., [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

  * LSTM 

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metr_la.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metr_la.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metr_la.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d pems_bay.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d pems_bay.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d pems_bay.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
    ```

  * TMeta-LSTM-GAL

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d metr_la.data.yml -p graph:Distance,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d metr_la.data.yml -p graph:Distance,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d metr_la.data.yml -p graph:Distance,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance,MergeIndex:12')
    ```

  * STMeta-V1

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:12')
    ```

  * STMeta-V2

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:12')
    ```

  * STMeta-V3

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d metr_la.data.yml -p graph:Distance-Correlation,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
              ' -d pems_bay.data.yml -p graph:Distance-Correlation,MergeIndex:12')
    ```

The results of METR-LA and PEMS-BAY can be found [here](./all_results.html#id7).