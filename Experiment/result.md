## Prediction Result


| Method | NYC | Chicago | DC |
| :-: | :-: | :-: | :-: |
|HM|6.79734|4.68078|3.66747|
|ARIMA|5.60477|3.79739|3.31826|
|HMM|5.42030|3.79743|3.20889|
|XGBoost|5.32069|3.75124|3.14101|
|LSTM|5.13589|3.68210|3.15595|
|Single Correlation Graph|4.77930|3.28816|2.86756|
|Single Distance Graph|4.78984|3.54810|2.76741|
|Single Interaction Graph|4.97007|3.23574|2.67335|
|Attention Based Graph Fusion|4.27751|3.09178|2.55618|

## 补充实验

#### （1）多层GCLSTM

City : DC

Method : Graph Fusion

lr: 5e-4

|    层数    | RMSE Remove Zero |
| :--------: | :--------------: |
| 一层GCLSTM |     2.55618      |
| 两层GCLSTM |     2.53276      |
| 三层GCLSTM |     2.53432      |

#### （2）Simple Average 和 Weighted Average

| Method                                                       |   NYC   | Chicago |   DC    |
| ------------------------------------------------------------ | :-----: | :-----: | :-----: |
| Naive Average 直接用三个single graph的结果进行平均           | 4.58827 | 3.22160 | 2.66711 |
| Weighted Naive Average 直接用三个single graph的结果进行加权平均 | 4.58469 | 3.18152 | 2.63383 |
| Hidden Feature Average 将GAL替换成Average                    |  -----  |  -----  |  -----  |
| Weighted hidden Feature Average 将GAL替换成参数加权Average   |  -----  |  -----  |  -----  |
| Attention based fusion                                       | 4.27751 | 3.09178 | 2.55618 |

#### （3）在一个Graph上去掉GCLSTM上的hidden state graph convolution

City : DC

Metric : RMSE remove zero

|          Method          | GCLSTM 去除 hidden state 上的GC | GCLSTM  |
| :----------------------: | :-----------------------------: | :-----: |
| Single Correlation Graph |             2.97511             | 2.86756 |
|  Single Distance Graph   |             2.97097             | 2.76741 |
| Single Interaction Graph |             2.90921             | 2.67335 |

#### ~~（4）调整lr~~

City : DC

Model : Attention Based Fusion

|  lr  | RMSE Remove Zero |
| :--: | :--------------: |
| 5e-5 |     2.65364      |
| 1e-4 |     2.60200      |
| 2e-4 |     2.58210      |
| 5e-4 |   **2.55618**    |
| 1e-3 |     2.51335      |

#### （5）训练数据长度

City : DC

Model : Attention Based Fusion

lr : 5e-4

| Train Day Length  | RMSERemove Zero |
| :---------------: | :-------------: |
| 完整数据集 40个月 |     2.55618     |
|      12个月       |     2.60002     |
|       6个月       |     2.65427     |
|       3个月       |     2.71175     |
