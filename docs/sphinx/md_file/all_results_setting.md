# Experiment Setting on different datasets

## STMeta Version

| Version Name | Temporal Feature Process | Temporal Merge Method | Multi-Graph Merge Method |
| :----------: | :----------------------: | :-------------------: | :----------------------: |
|  STMeta-V1   |          GCLSTM          |          GAL          |           GAL            |
|  STMeta-V2   |          GCLSTM          |     Concat&Dense      |           GAL            |
|  STMeta-V3   |          DCRNN           |          GAL          |           GAL            |

By default, we use `STMeta-V1` to run LSTM and single graph model tests.

## Search Space

|  Model  |                         Search Space                         |
| :-----: | :----------------------------------------------------------: |
|   HM    |               `CT: 0~6`, `PT: 0~7`, `TT: 0~4`                |
|  ARIMA  |                ``CT:168``,`p:3`, `d:0`, `q:0`                |
| XGBoost | `CT: 0~12`, `PT: 0~14`, `TT: 0~4`, `estimater: 10~200`, `depth: 2~10` |
|  GBRT   | `CT: 0~12`, `PT: 0~14`, `TT: 0~4`, `estimater: 10~200`, `depth: 2~10` |

## Results on Bike

### Dataset Statistics

|        Attributes        | **New York City** |   **Chicago**   |     **DC**      |
| :----------------------: | :---------------: | :-------------: | :-------------: |
|        Time span         |  2013.03-2017.09  | 2013.07-2017.09 | 2013.07-2017.09 |
| Number of riding records |    49,100,694     |   13,130,969    |   13,763,675    |
|    Number of stations    |        820        |       585       |       532       |

### Experiment Setting

* [ST_MGCN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/bike_trial.py) Run Code & Setting.

* [DCRNN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/bike_trial.py) Run Code & Setting.

* LSTM (C) & STMeta-V1  & STMeta-V2  & STMeta-V3

  All four models can be run by specifying data files and model files on  [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [bike_nyc.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/bike_nyc.data.yml) , [bike_chicago.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/bike_chicago.data.yml), [bike_dc.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/bike_dc.data.yml).

  * LSTM (C)

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    ```
    
    Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml).
    
  * STMeta-V1 
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    ```
  
    Model Files: [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).
  
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    ```
    
    
    Model Files: [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml).
    
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    ```
  
      Model Files: [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

### Experiment Results

|                 |        **NYC**         |       **Chicago**       |         **DC**          |
| :-------------: | :--------------------: | :---------------------: | :---------------------: |
|       HM        |    3.99224 `1-1-3`     |     2.97693 `1-1-2`     |     2.63165 `2-1-3`     |
|    ARIMA(C)     |  5.60928 `168-3-0-0`   |   3.83584 `168-3-0-0`   |   3.60450 `168-3-0-0`   |
|     XGBoost     | 4.12407 `12-10-0-20-5` |  2.92569 `9-7-0-20-5`   | 2.65671 `12-14-2-20-5`  |
|      GBRT       | 3.99907 `12-7-4-100-5` | 2.84257 `12-14-2-100-5` | 2.61768 `12-14-2-100-5` |
| ST_MGCN (G/DCI) |        3.72380         |         2.88300         |         2.48560         |
|  DCRNN(G/D C)   |        4.18666         |         3.27759         |         3.08616         |
|    LSTM (C)     |        4.55677         |         3.37004         |         2.91518         |
|    STMeta-V1    |        3.50475         |       **2.65511**       |         2.42582         |
|    STMeta-V2    |      **3.43870**       |         2.66379         |         2.41111         |
|    STMeta-V3    |        3.47834         |         2.66180         |       **2.38844**       |

## Results on DiDi

### Dataset Statistics

|        Attributes        |    **Xi'an**    |   **Chengdu**   |
| :----------------------: | :-------------: | :-------------: |
|        Time span         | 2016.10-2016.11 | 2016.10-2016.11 |
| Number of riding records |    5,922,961    |    8,439,537    |
|    Number of stations    |       256       |       256       |

### Experiment Setting

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
  }
  ```

  We can modify `city` parameter to `Chengdu` or `Xian` in [ST_ResNet.py]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_ResNet/ST_ResNet.py ) , and then run it.

  ```bash
   python ST_ResNet.py 
  ```

* [ST_MGCN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/didi_trial.py) Run Code & Setting.

* [DCRNN]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/didi_trial.py ) Run Code & Setting.

* LSTM (C) & STMeta-V1  & STMeta-V2  & STMeta-V3

  All four models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [didi_xian.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/didi_xian.data.yml) , [didi_chengdu.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/didi_chengdu.data.yml).

  * LSTM (C)

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
              ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    ```

    Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml).

  * STMeta-V1

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    
    ```
  
    Model Files: [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).
  
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
                '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
                '-p graph:Distance-Correlation-Interaction')   
    ```
  
    Model Files: [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml).
  
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
              '-p graph:Distance-Correlation-Interaction')
    ```
  
    Model Files: [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

### Experiment Results

|                   |       **Xi’an**        |      **Chengdu**      |
| :---------------: | :--------------------: | :-------------------: |
|        HM         |    6.18623 `1-1-2`     |    7.35477 `0-1-4`    |
|     ARIMA(C)      |  9.47478 `168-3-0-0`   | 12.52656 `168-3-0-0`  |
|      XGBoost      | 6.73346 `12-0-2-10-5 ` | 7.73836`9-14-4-20-2`  |
|       GBRT        | 6.44639 `9-0-2-50-2 `  | 7.58831 `12-7-2-50-5` |
|     ST-ResNet     |        6.08476         |        7.14638        |
|  ST_MGCN (G/DCI)  |        5.87456         |      **7.03217**      |
|   DCRNN(G/D C)    |        8.20254         |       11.50550        |
|     LSTM (C)      |        7.39970         |       10.11386        |
| STMeta-V1 (G/DCI) |        5.89154         |        7.06246        |
| STMeta-V2(G/DCI)  |      **5.75596**       |        7.09710        |
| STMeta-V3(G/DCI)  |        5.95507         |        7.04358        |

## Results on Metro

### Dataset Statistics

|        Attributes        |  **Chongqing**  |  **Shanghai**   |
| :----------------------: | :-------------: | :-------------: |
|        Time span         | 2016.08-2017.07 | 2016.07-2016.09 |
| Number of riding records |   409,277,117   |   333,149,034   |
|    Number of stations    |       113       |       288       |

### Experiment Setting

* [ST_MGCN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/metro_trial.py ) Run Code & Setting.

* [DCRNN](https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/metro_trial.py) Run Code & Setting.

* LSTM (C) & STMeta-V1  & STMeta-V2  & STMeta-V3

  All four models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [metro_chongqing.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/metro_chongqing.data.yml) , [metro_shanghai.data.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/metro_shanghai.data.yml).
  
  * LSTM (C) 

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
                ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml'
                ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    ```
  
    Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml).
  
  * STMeta-V1
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
                '-p graph:Distance-Correlation-Line')
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
                  '-p graph:Distance-Correlation-Line')
    ```
  
    Model Files: [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).
  
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line')
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_shanghai.data.yml '
            '-p graph:Distance-Correlation-Line')
    ```
  
  Model Files: [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml).
  
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
              '-p graph:Distance-Correlation-Line')
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
                '-p graph:Distance-Correlation-Line')
    ```
  

  Model Files: [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

### Experiment Results

|                   |       **Chongqing**       |       **Shanghai**        |
| :---------------: | :-----------------------: | :-----------------------: |
|        HM         |     120.30723 `0-1-4`     |     197.97092 `0-1-4`     |
|     ARIMA(C)      |   578.18563 `168-3-0-0`   |   792.1597 `168-3-0-0`    |
|      XGBoost      | 117.05069 `9-14-2-200-5`  |   185.00447`3-7-0-50-5`   |
|       GBRT        | 113.92276 1`2-10-4-200-5` | 186.74502 `12-10-0-100-2` |
|  ST_MGCN (G/DCI)  |         118.86668         |         181.55171         |
|   DCRNN(G/D C)    |         122.31121         |         326.97357         |
|     LSTM (C)      |        196.175732         |         368.8468          |
| STMeta-V1 (G/DCI) |       **92.74990**        |       **151.11746**       |
| STMeta-V2(G/DCI)  |         98.86152          |         158.21953         |
| STMeta-V3(G/DCI)  |         101.7806          |         156.58867         |

The period and trend features are more obvious in Metro dataset, so the performance is poor if only use closeness feature.

## Results on Charge-Station

### Dataset Statistics

|        Attributes        |   **Beijing**   |
| :----------------------: | :-------------: |
|        Time span         | 2018.03-2018.08 |
| Number of riding records |    1,272,961    |
|    Number of stations    |       629       |

### Experiment Setting

* [ST_MGCN]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/ST_MGCN/cs_trial.py ) Run Code & Setting.

* [DCRNN]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/DCRNN/cs_trial.py ) Run Code & Setting.

* LSTM (C) & STMeta-V1  & STMeta-V2  & STMeta-V3

  All four models can be run by specifying data files and model files on [STMeta_Obj.py](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_Obj.py).

  Data Files: [chargestation_beijing.data.yml]( https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/chargestation_beijing.data.yml).

  * LSTM (C) 

    ```python
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d chargestation_beijing.data.yml -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
    
    ```
    
    
    Model Files: [STMeta_v0.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v0.model.yml).
    
  * STMeta-V1
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
              ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
    
    ```
  
  
    Model Files: [STMeta_v1.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v1.model.yml).
  
  * STMeta-V2
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
                ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
    
    ```
  
  
    Model Files: [STMeta_v2.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v2.model.yml).
  
  * STMeta-V3
  
    ```python
    os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
                ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
    ```
  
  
    Model Files: [STMeta_v3.model.yml](https://github.com/Di-Chai/UCTB/blob/master/Experiments/STMeta/STMeta_v3.model.yml).

### Experiment Results

|                  |       **Beijing**       |
| :--------------: | :---------------------: |
|        HM        |     1.01610 `2-0-2`     |
|     ARIMA(C)     |   0.98236 `168-3-0-0`   |
|     XGBoost      |  0.83381 `12-7-0-20-2`  |
|       GBRT       | 0.82814 `12-10-0-100-2` |
|  ST_MGCN (G/DC)  |         0.82714         |
|   DCRNN(G/D C)   |         0.98871         |
|     LSTM (C)     |         1.58560         |
| STMeta-V1 (G/DC) |      **0.815518**       |
| STMeta-V2(G/DC)  |         0.82144         |
| STMeta-V3(G/DC)  |       **0.81541**       |

