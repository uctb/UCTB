# Stable Test Records

## DiDi Chengdu

#### Parameters for building graph

| Notation |          explanation           | value |
| :------: | :----------------------------: | :---: |
|    TD    |  threshold of distance graph   | 7500m |
|    TI    | threshold of interaction graph |  30   |
|    TC    | threshold of correlation graph | 0.65  |

#### Parameters for building model

```json
{
    "L": 1,
    "PT": 7,
    "lr": 5e-05,
    "TT": 4,
    "DenseUnits": 32,
    "GALUnits": 64,
    "LSTMUnits": 64,
    "ESlength": 500,
    "patience": 0.1,
    "Normalize": "True",
    "TI": 30.0,
    "CT": 6,
    "K": 1,
    "GALHeads": 2,
    "Graph": "Distance-Interaction-Correlation",
    "GLL": 1
}
```

|   实验编号   |   模型版本含义   |         Test-RMSE值          | Test-MAPE |
| :----------: | :--------------: | :--------------------------: | :-------: |
|      1       | AMulti-GCLSTM-V2 |           6.98410            |  0.35470  |
|      2       | AMulti-GCLSTM-V2 |           7.06971            |  0.36585  |
|      3       | AMulti-GCLSTM-V2 |           7.00403            |  0.34867  |
|      4       | AMulti-GCLSTM-V2 |           7.04557            |  0.34797  |
|      5       | AMulti-GCLSTM-V2 |           7.05717            |  0.36398  |
|      6       | AMulti-GCLSTM-V2 |           6.97287            |  0.34735  |
|      7       | AMulti-GCLSTM-V2 |           7.03885            |  0.35656  |
|      8       | AMulti-GCLSTM-V2 |           7.09894            |  0.36024  |
|      9       | AMulti-GCLSTM-V2 |           7.02147            |  0.33930  |
| 均值、标准差 |                  | 均值 7.03252，标准差 0.03865 |           |
|   平均耗时   |                  |          0.5h~1.5h           |           |

## DiDi Xian

```json
{
    "TrainDays": "All",
    "DenseUnits": 32,
    "GALUnits": 64,
    "Graph": "Distance-Interaction-Correlation",
    "CT": 6,
    "Train": "False",
    "Dataset": "DiDi",
    "GLL": 1,
    "TD": 7500.0,
    "GALHeads": 2,
    "patience": 0.1,
    "Epoch": 10000,
    "CodeVersion": "ST0",
    "TT": 4,
    "TC": 0.65,
    "Device": "1",
    "L": 1,
    "PT": 7,
    "ESlength": 500,
    "LSTMUnits": 64,
    "TI": 30.0,
    "Normalize": "True",
    "City": "Xian",
    "lr": 5e-05,
    "DataRange": "All",
    "BatchSize": 128,
    "K": 1,
    "Group": "Xian"
}
```

AMulti-GCLSTM-V2 多次实验结果，每次实验耗时 0.5h~1.5h

| 实验编号 | Test-RMSE | Test-MAPE |
| :------: | :-------: | :-------: |
|    1     |  5.80502  |  0.36022  |
|    2     |  5.88970  |  0.35590  |
|    3     |  6.00412  |  0.45126  |
|    4     |  5.93798  |  0.37956  |
|    5     |  6.01064  |  0.39242  |
|    6     |  5.89309  |  0.40803  |
|    7     |  5.84786  |  0.35915  |
|    8     |  5.88188  |  0.36777  |
|    9     |  5.97407  |  0.42393  |
|    10    |  5.80497  |  0.37014  |

最终结果：Test-RMSE 均值 5.90493，标准差 0.07142

Metro Shanghai

```json
{
    "TrainDays": "All",
    "patience": 0.1,
    "Train": "False",
    "TT": 4,
    "City": "ShanghaiV1",
    "ESlength": 500,
    "K": 1,
    "GLL": 1,
    "LSTMUnits": 64,
    "Normalize": "True",
    "PT": 7,
    "Epoch": 10000,
    "GALUnits": 64,
    "TI": 100.0,
    "lr": 2e-05,
    "Dataset": "Metro",
    "DenseUnits": 32,
    "L": 1,
    "Group": "Shanghai",
    "Graph": "Distance-line-Correlation",
    "DataRange": "All",
    "GALHeads": 2,
    "CodeVersion": "ST_Sim_0",
    "CT": 6,
    "TD": 5000.0,
    "TC": 0.7,
    "BatchSize": 128,
    "Device": "1"
}
```

AMulti-GCLSTM-V2 多次实验结果，每次实验耗时 6.5h~7.5h

| 实验编号 | Test-RMSE | Test-MAPE |
| :------: | :-------: | :-------: |
|    1     | 148.88104 |  0.13178  |
|    2     | 149.58350 |  0.14325  |
|    3     | 168.16162 |  0.14498  |
|    4     | 155.88750 |  0.19575  |
|    5     | 155.09171 |  0.18060  |
|    6     | 166.13303 |  0.18040  |
|    7     | 157.08799 |  0.15245  |

最终结果：Test-RMSE 均值 157.26091，标准差 6.90058

## ChargeStation Beijing

```json
{
    "GALUnits": 64,
    "TD": 1000.0,
    "TI": 500.0,
    "K": 1,
    "Train": "False",
    "CT": 6,
    "patience": 0.1,
    "ESlength": 200,
    "Graph": "Distance-Correlation",
    "Normalize": "True",
    "lr": 2e-05,
    "Device": "0",
    "BatchSize": 128,
    "LSTMUnits": 64,
    "City": "Beijing",
    "TrainDays": "All",
    "CodeVersion": "ST_Sim1_0",
    "TT": 4,
    "GALHeads": 2,
    "DenseUnits": 32,
    "PT": 7,
    "Group": "Beijing",
    "L": 1,
    "DataRange": "All",
    "TC": 0.1,
    "Epoch": 10000,
    "Dataset": "ChargeStation",
    "GLL": 1
}
```

AMulti-GCLSTM-V2 多次实验结果 (暂时只跑了4次)，每次实验耗时约 10h

| 实验编号 | Test-RMSE | Test-MAPE |
| :------: | :-------: | :-------: |
|    1     |  0.80954  |  0.22925  |
|    2     |  0.82956  |  0.23242  |
|    3     |  0.82393  |  0.22467  |
|    4     |  0.81360  |  0.22932  |

最终结果：Test-RMSE 均值 0.81915，标准差 0.0079745

