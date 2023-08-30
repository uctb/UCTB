#   Benchmark

## Datasets

| Application              |  Bike-sharing   |  Bike-sharing   |  Bike-sharing   |  Ride-sharing   |  Ride-sharing   |      Metro      |      Metro      |       EV        |  Traffic Speed  |  Traffic Speed  |
| :----------------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| City                     | *New York City* |    *Chicago*    |      *DC*       |     *Xi'an*     |    *Chengdu*    |   *Chongqing*   |   *Shanghai*    |    *Beijing*    |    *METR-LA*    |   *PEMS-BAY*    |
| Time span                | 2013.03-2017.09 | 2013.07-2017.09 | 2013.07-2017.09 | 2016.10-2016.11 | 2016.10-2016.11 | 2016.08-2017.07 | 2016.07-2016.09 | 2018.03-2018.05 | 2012.03-2012.06 | 2017.01-2017.07 |
| Number of riding records |   49,100,694    |   13,130,969    |   13,763,675    |    5,922,961    |    8,439,537    |   409,277,117   |   333,149,034   |    1,272,961    |     34,272      |     52,128      |
| Number of stations       |       820       |       585       |       532       |       256       |       256       |       113       |       288       |       629       |       207       |       325       |

Following shows the map-visualization of stations in NYC, Chicago, DC, Xian and Chengdu.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Bike_NYC.jpg" style="zoom:23%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Bike_Chicago.jpg" style="zoom:23%;height:800px;width:800px;"/> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Bike_DC.jpg" style="zoom:23%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/DiDi_Xian.jpg" style="zoom:23%;height:800px;width:800px;" />  <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/DiDi_Chengdu.jpg" style="zoom:23%;height:800px;width:800px;" />



Following shows the map-visualization of stations in Chongqing, Shanghai and Beijing, METR-LA and PEMS-BAY.

<img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Metro_Chongqing.jpg" style="zoom:23%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/Metro_Shanghai.jpg" style="zoom:23%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/EV_Beijing.jpg" style="zoom:23%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/METR_LA.png" style="zoom:23%;height:800px;width:800px;" /> <img src="https://uctb.github.io/UCTB/sphinx/md_file/src/image/PEMS_BAY.png" style="zoom:23%;height:800px;width:800px;" />  

## Results

We conducted experiments on the following datasets at the granularity of 15 minutes, 30 minutes, and 60 minutes respectively. More details and conclusions can be found in the this paper.  [IEEE Xplore](https://ieeexplore.ieee.org/document/9627543), [arXiv](https://arxiv.org/abs/2009.09379)

### 15-minute prediction tasks

The best two results are highlighted in bold, and the top one result is marked with `*'. (TC: Temporal Closeness; TM: Multi-Temporal Factors; SP: Spatial Proximity; SM: Multi-Spatial Factors; SD: Data-driven Spatial Knowledge Extraction

|                          |    NYC     |  Chicago   |     DC     |   Xi'an    |  Chengdu   |  Shanghai  | Chongqing  |  METR-LA   |  PEMS-BAY  |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| HM (TC)                  |   1.903    |   1.756    |   1.655    |   3.155    |   4.050    |   93.81    |   76.67    |   7.150    |   2.967    |
| ARIMA (TC)               |   1.874    |   1.784    |   1.689    |   3.088    |   3.948    |   83.54    |   67.11    |   7.028    |   2.869    |
| LSTM  (TC)               |   1.989    |   1.802    |   1.678    |   3.051    |   3.888    |   80.40    |   55.37    |   6.380    |   2.690    |
| HM (TM)                  |   1.892    |   1.668    |   1.555    |   2.828    |   3.347    |   49.75    |   45.26    |   8.934    |   3.690    |
| XGBoost (TM)             |   1.712    |   1.672    |   1.559    |   2.799    |   3.430    |   47.89    |   35.70    |   6.443    |   2.623    |
| GBRT (TM)                |   1.708    |   1.667    |   1.552    |   2.775    |   3.363    |   44.55    |   33.29    |   6.371    |   2.645    |
| TMeta-LSTM-GAL (TM)      |   1.818    |   1.623    |   1.540    |   2.917    |   3.286    |   45.88    |   33.34    |   6.156    |   2.544    |
| DCRNN (TC+SP)            |   1.712    |   1.718    |   1.594    |   2.889    |   3.743    |   56.00    |   37.07    |   6.440    |   5.322    |
| STGCN (TC+SP)            |   1.738    |   1.806    |   1.630    |   2.789    |   3.453    |   47.40    |   35.19    |   6.236    |   2.493    |
| GMAN (TC+SP)             | **1.632*** | **1.529**  | **1.355*** |   2.769    |   3.520    |   49.21    |   36.66    |   6.214    |   3.484    |
| Graph-WaveNet (TC+SP+SD) | **1.644**  | **1.460*** | **1.357**  |   2.764    |   3.442    |   47.84    |   35.04    | **5.270*** |   2.780    |
| ST-ResNet (TM+SP)        |    ---     |    ---     |    ---     |   2.686    |   3.314    |    ---     |    ---     |    ---     |    ---     |
| ST-MGCN (TM+SM)          |   1.687    |   1.646    |   1.545    |   2.714    |   3.293    |   46.54    | **32.72**  |   6.645    | **2.426*** |
| AGCRN-CDW (TM+SD)        |   1.836    |   1.883    |   1.745    |   2.722    |   3.296    |   77.06    |   46.95    |   6.709    |   2.453    |
| STMeta-GCL-GAL (TM+SM)   |   1.659    |   1.607    |   1.527    |   2.653    | **3.244**  | **41.67**  | **31.39*** | **5.644**  | **2.433**  |
| STMeta-GCL-CON (TM+SM)   |   1.673    |   1.629    |   1.512    | **2.637*** | **3.241*** |   43.83    |   38.21    |   5.800    |   2.449    |
| STMeta-DCG-GAL (TM+SM)   |   1.654    |   1.609    |   1.517    | **2.648**  |   3.254    | **40.94*** |   36.90    |   5.788    |   2.446    |

### Results on 30-minute prediction tasks

The best two results are highlighted in bold, and the top one result is marked with `*'. (TC: Temporal Closeness; TM: Multi-Temporal Factors; SP: Spatial Proximity; SM: Multi-Spatial Factors; SD: Data-driven Spatial Knowledge Extraction

|                          |    NYC     |  Chicago   |     DC     |   Xi'an    |  Chengdu   |  Shanghai  | Chongqing  |  Beijing   |  METR-LA   |  PEMS-BAY  |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| HM (TC)                  |   3.206    |   2.458    |   2.304    |   5.280    |   6.969    |   269.16   |   221.39   |   0.768    |   9.471    |   4.155    |
| ARIMA (TC)               |   3.178    |   2.428    |   2.228    |   5.035    |   6.618    |   212.01   |   180.53   |   0.755    |   9.230    |   3.936    |
| LSTM  (TC)               |   3.018    |   2.493    |   2.212    |   4.950    |   6.444    |   195.60   |   104.61   |   0.755    |   7.866    |   3.683    |
| HM (TM)                  |   2.686    |   2.230    |   1.956    |   4.239    |   4.851    |   108.59   |   74.55    |   0.864    |   9.560    |   3.965    |
| XGBoost (TM)             |   2.704    |   2.376    |   1.956    |   4.172    |   4.915    |   81.82    |   69.50    |   0.686    |   8.298    |   3.253    |
| GBRT (TM)                |   2.682    |   2.355    |   1.928    |   4.135    |   4.873    |   83.94    |   72.99    |   0.689    |   8.269    |   3.370    |
| TMeta-LSTM-GAL (TM)      |   2.511    | **2.133*** |   1.927    |   3.847    |   4.678    |   85.19    |   53.18    |   0.686    |   7.436    |   3.231    |
| DCRNN (TC+SP)            |   2.618    |   2.246    |   2.118    |   4.529    |   6.258    |   116.15   |   65.72    |   0.757    |   8.562    |   6.198    |
| STGCN (TC+SP)            |   2.841    |   2.482    |   2.067    |   3.992    |   5.051    |   91.29    |   58.34    |   0.694    |   7.871    |   3.136    |
| GMAN (TC+SP)             |   2.792    |   2.336    | **1.836*** |   4.026    |   5.293    |   97.58    |   51.37    |   0.764    |   7.276    |   3.688    |
| Graph-WaveNet (TC+SP+SD) |   2.666    |   2.158    |   1.874    |   3.986    |   5.097    |   92.88    |   52.52    |   0.719    | **6.809*** |   3.589    |
| ST-ResNet (TM+SP)        |    ---     |    ---     |    ---     |   3.903    |   4.673    |    ---     |    ---     |    ---     |    ---     |    ---     |
| ST-MGCN (TM+SM)          |   2.513    |   2.177    |   1.903    |   3.886    |   4.732    |   88.76    |   50.96    |   0.691    |   8.079    | **3.042**  |
| AGCRN-CDW (TM+SD)        |   2.830    |   2.565    |   2.074    |   3.958    |   4.753    |   238.99   |   131.55   |   0.688    |   8.575    | **3.022*** |
| STMeta-GCL-GAL (TM+SM)   | **2.410*** |   2.170    |   1.856    | **3.808**  |   4.650    | **75.36*** | **49.47**  | **0.670**  |   7.156    |   3.116    |
| STMeta-GCL-CON (TM+SM)   | **2.411**  | **2.133*** |   1.859    | **3.772*** | **4.613*** |   80.69    |   50.01    | **0.667*** | **6.889*** |   3.204    |
| STMeta-DCG-GAL (TM+SM)   | **2.411**  |   2.182    | **1.852**  |   3.833    | **4.635**  | **77.49**  | **48.96*** | **0.670**  |   7.184    |   3.187    |

### Results on 60-minute prediction tasks

The best two results are highlighted in bold, and the top one result is marked with `*'. (TC: Temporal Closeness; TM: Multi-Temporal Factors; SP: Spatial Proximity; SM: Multi-Spatial Factors; SD: Data-driven Spatial Knowledge Extraction

|                          |    NYC     |  Chicago   |     DC     |   Xi'an    |  Chengdu   |  Shanghai   | Chongqing  |  Beijing   |  METR-LA   |  PEMS-BAY  |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :---------: | :--------: | :--------: | :--------: | :--------: |
| HM (TC)                  |   5.814    |   4.143    |   3.485    |   10.136   |   14.145   |   824.94    |   673.55   |   1.178    |   12.303   |   5.779    |
| ARIMA (TC)               |   5.289    |   3.744    |   3.183    |   9.475    |   13.259   |   676.79    |   578.19   |   0.982    |   11.739   |   5.670    |
| LSTM  (TC)               |   5.167    |   3.721    |   3.234    |   9.830    |   13.483   |   506.07    |   322.81   |   0.999    |   10.083   |   4.777    |
| HM (TM)                  |   3.992    |   3.104    |   2.632    |   6.186    |   7.512    |   172.55    |   119.86   |   1.016    |   10.727   |   4.018    |
| XGBoost (TM)             |   4.102    |   3.003    |   2.643    |   6.733    |   7.592    |   160.38    |   117.05   |   0.834    |   10.299   |   3.703    |
| GBRT (TM)                |   4.039    |   2.984    |   2.611    |   6.446    |   7.511    |   154.29    |   113.92   |   0.828    |   10.013   |   3.704    |
| TMeta-LSTM-GAL (TM)      |   3.739    |   2.840    |   2.557    | **5.843**  |   6.949    |   163.31    |   102.86   |   0.840    | **8.670*** |   3.616    |
| DCRNN (TC+SP)            |   4.187    |   3.081    |   3.016    |   8.203    |   11.444   |   340.25    |   122.31   |   0.989    |   11.121   |   6.920    |
| STGCN (TC+SP)            |   3.895    |   2.989    |   2.597    |   6.150    |   7.710    |   187.98    |   106.16   |   0.859    |   10.688   |   3.472    |
| GMAN (TC+SP)             |   4.251    |   2.875    |   2.530    |   7.099    |   13.351   |   193.39    |   117.52   |   0.949    |   10.012   |   3.846    |
| Graph-WaveNet (TC+SP+SD) |   3.863    |   2.812    | **2.403*** |   6.541    |   8.162    |   186.82    |   102.75   |   0.930    |   9.463    |   4.135    |
| ST-ResNet (TM+SP)        |    ---     |    ---     |    ---     |   6.075    |   7.155    |     ---     |    ---     |    ---     |    ---     |    ---     |
| ST-MGCN (TM+SM)          |   3.723    |   2.904    |   2.518    |   5.878    |   7.067    |   159.52    |   104.87   |   0.827    |   10.798   | **3.486**  |
| AGCRN-CDW (TM+SD)        |   3.795    |   2.935    |   2.580    |   8.835    |   10.275   |   658.12    |   287.41   |   0.844    |   10.728   | **3.381*** |
| STMeta-GCL-GAL (TM+SM)   | **3.518**  | **2.695**  |   2.405    |   5.871    | **6.858*** |   153.17    | **97.87**  |   0.831    | **8.834**  |   3.514    |
| STMeta-GCL-CON (TM+SM)   | **3.507*** |   2.739    | **2.404**  | **5.829*** | **6.873**  | **149.05**  |   106.41   | **0.807**  |   9.147    |   3.552    |
| STMeta-DCG-GAL (TM+SM)   |   3.521    | **2.652*** |   2.423    |   5.908    |   6.904    | **143.18*** | **94.78*** | **0.803*** |   8.993    |   3.500    |

## Implement Details

### Search Space

We use [nni](https://github.com/microsoft/nni) toolkit to search the best parameters of HM, XGBoost and GBRT model. Search space are following.

|  Model  |                         Search Space                         |
| :-----: | :----------------------------------------------------------: |
|   HM    |               `CT: 0~6`, `PT: 0~7`, `TT: 0~4`                |
|  ARIMA  |                ``CT:168``,`p:3`, `d:0`, `q:0`                |
| XGBoost | `CT: 0~12`, `PT: 0~14`, `TT: 0~4`, `estimater: 10~200`, `depth: 2~10` |
|  GBRT   | `CT: 0~12`, `PT: 0~14`, `TT: 0~4`, `estimater: 10~200`, `depth: 2~10` |

### Bike-sharing

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

### Ride-sharing

* HM & XGBoost & GBRT

  | 15 minutes |   **Xi'an**   |  **Chengdu**   |
  | :--------: | :-----------: | :------------: |
  |     HM     |    `5-0-4`    |    `2-7-4`     |
  |  XGBoost   | `7-14-0-10-4` | `12-14-1-27-3` |
  |    GBRT    | `11-2-2-45-3` | `13-15-5-39-3` |

  | 30 minutes |  **Xi'an**   |  **Chengdu**   |
  | :--------: | :----------: | :------------: |
  |     HM     |   `2-0-2`    |    `1-7-4`     |
  |  XGBoost   | `9-0-2-25-3` | `9-14-3-16-3`  |
  |    GBRT    | `9-0-2-80-3` | `10-10-5-34-3` |

  | 60 minutes |   **Xi'an**   |  **Chengdu**  |
  | :--------: | :-----------: | :-----------: |
  |     HM     |    `1-1-2`    |    `0-7-4`    |
  |  XGBoost   | `12-0-2-10-5` | `9-6-2-14-3`  |
  |    GBRT    | `9-0-2-50-2`  | `9-12-2-50-5` |

* ST-ResNet

  |                    ST-ResNet Search Space                    |
  | :----------------------------------------------------------: |
  | ``residual_units:2~6``,  `conv_filter:[32, 64, 128]`,  `kernal_size:3~5`, <br />`lr:[0.0001, 0.00002, 0.00004, 0.00008, 0.00001]`, `batch_size:[32, 64, 128, 256]` |

  The best parameters found are following.

  ```python
  args = {
    'dataset': 'DiDi',
    'city': 'Chengdu',
    'num_residual_unit': 4,
    'conv_filters': 64,
    'kernel_size': 3,
    'lr': 1e-5,
    'batch_size': 32
  
  ```

  We can modify `city` parameter to `Chengdu` or `Xian` in [ST_ResNet.py]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_ResNet/ST_ResNet.py ) , and then run it.

  ```bash
   python ST_ResNet.py 
  ```

* [ST_MGCN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/didi_trial.py) Run Code & Setting.

* [DCRNN]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/didi_trial.py ) Run Code & Setting.

* LSTM & TMeta-LSTM-GAL & STMeta-V1  & STMeta-V2  & STMeta-V3

  These five models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [didi_xian.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/didi_xian.data.yml) , [didi_chengdu.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/didi_chengdu.data.yml).

  Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml), [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).,  [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml)., [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

  * LSTM
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
            ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
            ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')
    ```
  
  * TMeta-LSTM-GAL
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml -p graph:Distance,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml -p graph:Distance,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml -p graph:Distance,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance,MergeIndex:12')
    ```
  
  * STMeta-V1
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
    ```
    
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
    ```
  
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
    
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction,MergeIndex:12')
    ```
    

### Metro Passenger

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
    

### Electric Vehicle

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

### Traffic Speed

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

