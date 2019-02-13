## GCN Result

#### RMSE remove zero


| Method | NYC | Chicago | DC |
| :-: | :-: | :-: | :-: |
|HM|6.79734|4.68078|3.66747|
|ARIMA|5.60477|3.79739|3.31826|
|HMM|5.42030|3.79743|3.20889|
|XGBoost|5.32069|3.75124|3.14101|
|LSTM|5.13589|3.68210|3.15595|
|Correlation Graph|4.77930|3.28816|2.86313|
|Distance Graph|4.78984|3.54810|2.84960|
|Interaction Graph|4.97007|3.23574|2.77634|
|Graph Fusion|4.27751|3.09178|2.65364|



#### Correlation-Graph Metric(RMSE remove zero)

| GCN Parmeter |   NYC   | Chicago |   DC    |
| :----------: | :-----: | :-----: | :-----: |
|     K0L1     | 5.15054 | 3.16184 | 3.71221 |
|     K1L1     | 4.63072 | 2.83749 | 3.31452 |
|     K2L1     | 4.65754 | 2.88097 | 3.32182 |
|     K3L1     | 4.65675 | 2.83347 | 3.40716 |

#### RMSE with zero

|      Method       |   NYC   | Chicago |   DC    |
| :---------------: | :-----: | :-----: | :-----: |
|        HM         | 5.39427 | 3.25643 | 2.58532 |
|       ARIMA       | 4.33433 | 2.54833 | 2.23166 |
|        HMM        | 4.18800 | 2.53786 | 2.16349 |
|      XGBoost      | 4.09450 | 2.49430 | 2.09572 |
|       LSTM        | 4.00914 | 2.51368 | 2.17168 |
| Correlation Graph | 3.71546 | 2.25233 | 1.96832 |
|  Distance Graph   | 3.73550 | 2.40881 | 1.94557 |
| Interaction Graph | 3.93057 | 2.19883 | 1.89554 |
|   Graph Fusion    | 3.32595 | 2.11923 | 1.83227 |

## 补充实验设计

#### (1) 验证GCLSTM的有效性

GCLSTM和直接堆叠GCN+LSTM的区别是多了一个在hidden state上的GCN，可以实验把hidden state上的GCN去掉，看看结果的变化

结果整理：(one city)

|          Method          | GCLSTM | GCLSTM(Removed GC on hidden state) |
| :----------------------: | :----: | :--------------------------------: |
| Single Correlation Graph |        |                                    |
|  Single Distance Graph   |        |                                    |
| Single Interaction Graph |        |                                    |
|       Graph Fusion       |        |                                    |

#### (2) 验证Graph Fusion的有效性

选取一些基础的Graph Fusion方法进行实验

结果整理 (one city)

| Method                                                       |      |
| ------------------------------------------------------------ | ---- |
| Naive Average 直接用三个single graph的结果进行平均           |      |
| Hidden Feature Average 在现有模型中 将GAL替换成Average       |      |
| Weighted hidden Feature Average 在现有模型中 将GAL替换成参数加权Average |      |
| Attention based fusion                                       |      |

#### (3) 验证Graph的作用

目前构建Correlation Graph时，站点间的Pearson coefficient大于0，就进行连接，可以调整这个阈值进行实验，调整为: -1, -0.5, 0, 0.5 四个值（0 为现在展示的结果，-1即为全连接图）

## 实际实验

#### （1）GCLSTM两层

#### （2）Simple Average 和 Weighted Average

#### （3）在一个Graph上去掉GCLSTM上的hidden state GCN