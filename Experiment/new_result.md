## Baseline

| Method  |   NYC   | Chicago |   DC    |
| :-----: | :-----: | :-----: | :-----: |
|   HM    | 6.79734 | 4.68078 | 3.66747 |
|  ARIMA  | 5.60477 | 3.79739 | 3.31826 |
|   HMM   | 5.42030 | 3.79743 | 3.20889 |
| XGBoost | 5.32069 | 3.75124 | 3.14101 |
|  LSTM   | 5.13589 | 3.68210 | 3.15595 |

## AMulti-GCLSTM

#### Experiment 1

```python
# Experiment Setting
{
    "TD": "1000",
    "L": "1",
    "TrainDays": "All",
    "TI": "500",
    "GALUnits": "64",
    "TC": "0",
    "Graph": "Interaction",
    "Epoch": "10000",
    "CodeVersion": "V0",
    "LSTMUnits": "64",
    "T": "6",
    "K": "1",
    "lr": "1e-3",
    "GLL": "1",
    "GALHeads": "2",
    "BatchSize": "64",
    "City": "NYC",
    "patience": "50",
    "DenseUnits": "32"
}
```

|         SG Types         |   NYC   | Chicago |   DC    |
| :----------------------: | :-----: | :-----: | :-----: |
| Single Correlation Graph | 4.26118 | 3.07176 | 2.74649 |
|  Single Distance Graph   | 4.30532 | 2.93696 | 2.54211 |
| Single Interaction Graph | 3.88593 | 2.68048 | 2.45289 |
|      AMulti-GCLSTM       | 3.33784 | 2.56888 | 2.27439 |

#### Experiment 2

```python
# Experiment Setting
{
    "TD": "1000",
    "L": "1",
    "TrainDays": "All",
    "TI": "500",
    "GALUnits": "64",
    "TC": "0",
    "Graph": "Interaction",
    "Epoch": "10000",
    "CodeVersion": "V0",
    "LSTMUnits": "64",
    "T": "12",
    "K": "1",
    "lr": "1e-3",
    "GLL": "1",
    "GALHeads": "2",
    "BatchSize": "64",
    "City": "NYC",
    "patience": "50",
    "DenseUnits": "32"
}
```

|         SG Types         |   NYC   | Chicago |   DC    |
| :----------------------: | :-----: | :-----: | :-----: |
| Single Correlation Graph | 4.02801 | 2.84327 | 2.62097 |
|  Single Distance Graph   | 3.76683 | 2.67796 | 2.42099 |
| Single Interaction Graph | 3.74241 | 2.71236 | 2.41558 |
|      AMulti-GCLSTM       | 3.24510 | 2.36098 | 2.19385 |