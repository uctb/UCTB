## Base experiment result

#### Pre-train result

|    Graph    |  City   | Result  |
| :---------: | :-----: | :-----: |
| Correlation |   NYC   | 7.00824 |
| Correlation | Chicago | 3.29530 |
| Correlation |   DC    | 2.86770 |
|  Distance   |   NYC   | 7.96931 |
|  Distance   | Chicago | 3.10245 |
|  Distance   |   DC    | 3.78748 |

#### Parameters (chosen by experiments)

1. FT与Transfer，Validate数据 使用训练数据的30%，Test data 长度约为30天
2. TD的训练数据为工作日（周三）
3. 训练学习率 1e-5，early-stopping的测试长度为500-epoch （当lr较大时FT效果会很差，所以保持了比较小的值，lr对Transfer的影响不大）
4. static training

|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Correlation|checkin|nyc|chicago|0.1|1天|4.67712|**3.92269**|3.99061|
|Correlation|checkin|nyc|chicago|0.1|3天|4.22121|**3.71275**|3.74433|
|Correlation|checkin|nyc|chicago|0.1|5天|4.32650|**3.87218**|4.06787|
|Correlation|checkin|nyc|chicago|0.1|7天|3.77726|**3.62829**|3.64411|
|Correlation|checkin|nyc|dc|0.1|1天|3.77775|3.41202|**3.24368**|
|Correlation|checkin|nyc|dc|0.1|3天|3.53591|3.23791|**3.22884**|
|Correlation|checkin|nyc|dc|0.1|5天|3.39463|3.39410|**3.39393**|
|Correlation|checkin|nyc|dc|0.1|7天|3.38310|3.23871|**3.23573**|
|Correlation|checkin|chicago|dc|0.1|1天|3.78157|3.71789|**3.69317**|
|Correlation|checkin|chicago|dc|0.1|3天|3.69740|**3.13670**|3.16728|
|Correlation|checkin|chicago|dc|0.1|5天|3.62814|**3.16764**|3.18577|
|Correlation|checkin|chicago|dc|0.1|7天|3.58433|**3.11301**|3.15291|
|Correlation|checkin|chicago|nyc|0.1|1天|7.33929|**6.90001**|6.96449|
|Correlation|checkin|chicago|nyc|0.1|3天|7.22262|6.79849|**6.76211**|
|Correlation|checkin|chicago|nyc|0.1|5天|7.13852|**6.69655**|7.10062|
|Correlation|checkin|chicago|nyc|0.1|7天|7.12167|**6.69928**|7.10788|
|Correlation|checkin|dc|nyc|0.1|1天|9.42764|7.78120|**7.75579**|
|Correlation|checkin|dc|nyc|0.1|3天|7.92233|**6.91918**|8.86747|
|Correlation|checkin|dc|nyc|0.1|5天|7.87232|**7.59995**|7.87427|
|Correlation|checkin|dc|nyc|0.1|7天|7.81039|**7.43801**|7.81107|
|Correlation|checkin|dc|chicago|0.1|1天|10.58203|**4.67753**|4.75832|
|Correlation|checkin|dc|chicago|0.1|3天|11.23320|4.37880|**4.00140**|
|Correlation|checkin|dc|chicago|0.1|5天|9.93056|3.78431|**3.73602**|
|Correlation|checkin|dc|chicago|0.1|7天|8.54751|4.07984|**3.84819**|

|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Distance|checkin|nyc|chicago|0.1|1天|4.70441|**4.70361**|12.59841|
|Distance|checkin|nyc|chicago|0.1|3天|4.70441|4.82347|4.78770|
|Distance|checkin|nyc|chicago|0.1|5天|**4.70441**|4.83354|4.81036|
|Distance|checkin|nyc|chicago|0.1|7天|4.70441|**4.70070**|4.74429|
|Distance|checkin|nyc|dc|0.1|1天|4.18998|**4.18815**|4.18669|
|Distance|checkin|nyc|dc|0.1|3天|4.18998|3.95583|**3.95511**|
|Distance|checkin|nyc|dc|0.1|5天|4.18998|**3.96894**|4.01603|
|Distance|checkin|chicago|dc|0.1|1天|3.19900|**3.10418**|3.11494|
|Distance|checkin|chicago|dc|0.1|3天|3.19900|**3.00662**|3.13352|
|Distance|checkin|chicago|dc|0.1|5天|3.19900|**2.98474**|2.99215|
|Distance|checkin|chicago|dc|0.1|7天|3.19900|**2.89581**|2.94413|
|Distance|checkin|chicago|nyc|0.1|1天|10.24455|**9.44282**|12.85142|
|Distance|checkin|chicago|nyc|0.1|3天|10.24455|**6.87987**|12.93270|
|Distance|checkin|chicago|nyc|0.1|5天|10.24455|**10.21737**|13.56898|
|Distance|checkin|chicago|nyc|0.1|7天|10.24455|**6.66069**|12.12317|
|Distance|checkin|dc|nyc|0.1|1天|7.23524|7.23436|**7.23350**|
|Distance|checkin|dc|nyc|0.1|3天|7.23524|**6.73650**|6.81929|
|Distance|checkin|dc|nyc|0.1|5天|7.23524|**7.23440**|7.23744|
|Distance|checkin|dc|nyc|0.1|7天|**7.23524**|7.23836|7.23745|
|Distance|checkin|dc|chicago|0.1|1天|**4.34642**|4.35396|4.34736|
|Distance|checkin|dc|chicago|0.1|5天|**4.34642**|4.49737|4.39714|
|Distance|checkin|dc|chicago|0.1|7天|**4.34642**|4.49232|4.37209|
|Distance|checkin|dc|chicago|0.1|3天|**4.34642**|4.46849|4.47618|

总结

1. Correlation graph 的迁移效果比 Distance graph 好
2. 迁移效果较明显的是：NYC => DC，DC => Chicago

## Tuning match methods

transfer ratio = 0.1，graph = Correlation，其余参数与上一节列出的相同

以下为使用不用match方法进行匹配的迁移结果，结果展示以迁移方向为组、每组内包含4种相似站点匹配方法，加粗的是目标区域有1天3天的情况下，迁移结果rmse最小的匹配方法

1. checkin 匹配方法：使用站点附近1km、工作日与节假日的checkin特征进行匹配
2. poi 匹配方法：使用站点附近1km的poi特征进行匹配
3. traffic：使用目标区域训练数据进行匹配
4. fake_traffic：在目标区域使用一个月流量数据进行匹配，训练数据、测试数据保持与其他实验一致

|      Match       |   SD    |   TD    | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :--------------: | :-----: | :-----: | :-------------: | :-------: | :-----: | :---------: |
|     checkin      |   nyc   | chicago |       1天       |  4.76519  | 3.91775 |   4.00033   |
|     checkin      |   nyc   | chicago |       3天       |  4.21627  | 3.71755 |   3.73617   |
|     **poi**      |   nyc   | chicago |     **1天**     |  4.76519  | 3.91775 | **3.99740** |
|     **poi**      |   nyc   | chicago |     **3天**     |  4.21627  | 3.71755 | **3.71152** |
|     traffic      |   nyc   | chicago |       1天       |  4.76519  | 3.91775 |   3.99903   |
|     traffic      |   nyc   | chicago |       3天       |  4.21627  | 3.71755 |   3.71532   |
|   fake_traffic   |   nyc   | chicago |       1天       |  4.76519  | 3.91775 |   4.01753   |
|   fake_traffic   |   nyc   | chicago |       3天       |  4.21627  | 3.71755 |   3.71876   |
|     checkin      |   nyc   |   dc    |       1天       |  3.77953  | 3.40948 |   3.27021   |
|     checkin      |   nyc   |   dc    |       3天       |  3.53572  | 3.24863 |   3.22698   |
|       poi        |   nyc   |   dc    |       1天       |  3.77953  | 3.40948 |   3.26955   |
|       poi        |   nyc   |   dc    |       3天       |  3.53572  | 3.24863 |   3.23642   |
|     traffic      |   nyc   |   dc    |       1天       |  3.77953  | 3.40948 |   3.27314   |
|     traffic      |   nyc   |   dc    |       3天       |  3.53572  | 3.24863 |   3.24170   |
| **fake_traffic** |   nyc   |   dc    |     **1天**     |  3.77953  | 3.40948 | **3.25348** |
| **fake_traffic** |   nyc   |   dc    |     **3天**     |  3.53572  | 3.24863 | **3.23603** |
|     checkin      | chicago |   dc    |       1天       |  3.78083  | 3.71714 |   3.69250   |
|   **checkin**    | chicago |   dc    |     **3天**     |  3.69748  | 3.15200 | **3.15129** |
|     **poi**      | chicago |   dc    |     **1天**     |  3.78083  | 3.71714 | **3.69245** |
|       poi        | chicago |   dc    |       3天       |  3.69748  | 3.15200 |   3.16164   |
|     traffic      | chicago |   dc    |       1天       |  3.78083  | 3.71714 |   3.68980   |
|     traffic      | chicago |   dc    |       3天       |  3.69748  | 3.15200 |   3.18018   |
|   fake_traffic   | chicago |   dc    |       1天       |  3.78083  | 3.71714 |   3.70910   |
|   fake_traffic   | chicago |   dc    |       3天       |  3.69748  | 3.15200 |   3.20673   |
|     checkin      | chicago |   nyc   |       1天       |  7.33863  | 6.90045 |   6.96378   |
|   **checkin**    | chicago |   nyc   |     **3天**     |  7.22249  | 6.83481 | **6.77691** |
|       poi        | chicago |   nyc   |       1天       |  7.33863  | 6.90045 |   6.94736   |
|       poi        | chicago |   nyc   |       3天       |  7.22249  | 6.83481 |   6.80737   |
|     traffic      | chicago |   nyc   |       1天       |  7.33863  | 6.90045 |   6.96541   |
|     traffic      | chicago |   nyc   |       3天       |  7.22249  | 7.00043 |   7.78913   |
| **fake_traffic** | chicago |   nyc   |     **1天**     |  7.33863  | 6.90045 | **6.94016** |
|   fake_traffic   | chicago |   nyc   |       3天       |  7.22249  | 6.83481 |   6.81975   |
|   **checkin**    |   dc    |   nyc   |     **1天**     |  9.38367  | 7.77615 | **7.49206** |
|   **checkin**    |   dc    |   nyc   |     **3天**     |  7.92180  | 6.93686 | **7.64479** |
|       poi        |   dc    |   nyc   |       1天       |  9.38367  | 7.77615 |   7.54143   |
|       poi        |   dc    |   nyc   |       3天       |  7.92180  | 6.93686 |   7.92671   |
|     traffic      |   dc    |   nyc   |       1天       |  9.38367  | 7.77615 |   7.58968   |
|     traffic      |   dc    |   nyc   |       3天       |  7.92180  | 6.93686 |   8.59754   |
|   fake_traffic   |   dc    |   nyc   |       1天       |  9.38367  | 7.77615 |   7.72215   |
|   fake_traffic   |   dc    |   nyc   |       3天       |  7.92180  | 6.93686 |   9.24880   |
|     checkin      |   dc    | chicago |       1天       | 10.60726  | 4.71315 |   4.52085   |
|   **checkin**    |   dc    | chicago |     **3天**     | 11.21222  | 3.79817 | **3.58828** |
|     **poi**      |   dc    | chicago |     **1天**     | 10.60726  | 4.71315 | **4.48266** |
|       poi        |   dc    | chicago |       3天       | 11.21222  | 3.79817 |   3.81154   |
|     traffic      |   dc    | chicago |       1天       | 10.60726  | 4.71315 |   4.52752   |
|     traffic      |   dc    | chicago |       3天       | 11.21222  | 3.79817 |   3.64937   |
|   fake_traffic   |   dc    | chicago |       1天       | 10.60726  | 4.71315 |   4.70475   |
|   fake_traffic   |   dc    | chicago |       3天       | 11.21222  | 3.79817 |   3.67280   |

## Tuning  transfer mode

graph：correlation，transfer ratio：0.1，match-method：checkin

| TrainMode |   SD    |   TD    | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :-------: | :-----: | :-----: | :-------------: | :-------: | :-----: | :---------: |
|  dynamic  |   nyc   | chicago |       1天       |  4.67712  | 3.92269 |   3.99061   |
|  static   |   nyc   | chicago |       1天       |  4.67712  | 3.92269 |   3.99061   |
|  dynamic  |   nyc   | chicago |       3天       |  4.22121  | 3.72731 |   3.72169   |
|  static   |   nyc   | chicago |       3天       |  4.22121  | 3.72731 |   3.74433   |
|  dynamic  |   nyc   | chicago |       5天       |  4.32650  | 4.03527 |   4.09383   |
|  static   |   nyc   | chicago |       5天       |  4.32650  | 4.03527 |   4.06787   |
|  dynamic  |   nyc   | chicago |       7天       |  3.77726  | 3.62886 |   3.64174   |
|  static   |   nyc   | chicago |       7天       |  3.77726  | 3.62886 |   3.64411   |
|  dynamic  |   nyc   |   dc    |       1天       |  3.77775  | 3.41219 |   3.24365   |
|  static   |   nyc   |   dc    |       1天       |  3.77775  | 3.41219 |   3.24368   |
|  dynamic  |   nyc   |   dc    |       3天       |  3.53591  | 3.24496 |   3.23717   |
|  static   |   nyc   |   dc    |       3天       |  3.53591  | 3.24496 |   3.22884   |
|  dynamic  |   nyc   |   dc    |       5天       |  3.39463  | 3.39877 |   3.41054   |
|  static   |   nyc   |   dc    |       5天       |  3.39463  | 3.39877 |   3.39393   |
|  dynamic  |   nyc   |   dc    |       7天       |  3.38310  | 3.29157 |   3.27758   |
|  static   |   nyc   |   dc    |       7天       |  3.38310  | 3.29157 |   3.23573   |
|  dynamic  | chicago |   dc    |       1天       |  3.78157  | 3.71789 |   3.69317   |
|  static   | chicago |   dc    |       1天       |  3.78157  | 3.71789 |   3.69317   |
|  dynamic  | chicago |   dc    |       3天       |  3.69740  | 3.14631 |   3.18608   |
|  static   | chicago |   dc    |       3天       |  3.69740  | 3.14631 |   3.16728   |
|  dynamic  | chicago |   dc    |       5天       |  3.62814  | 3.17078 |   3.19249   |
|  static   | chicago |   dc    |       5天       |  3.62814  | 3.17078 |   3.18577   |
|  dynamic  | chicago |   dc    |       7天       |  3.58433  | 3.11549 |   3.15131   |
|  static   | chicago |   dc    |       7天       |  3.58433  | 3.11549 |   3.15291   |
|  dynamic  | chicago |   nyc   |       1天       |  7.33929  | 7.38597 |   6.96449   |
|  static   | chicago |   nyc   |       1天       |  7.33929  | 7.38597 |   6.96449   |
|  dynamic  | chicago |   nyc   |       3天       |  7.55972  | 6.80038 |   6.77189   |
|  static   | chicago |   nyc   |       3天       |  7.22262  | 6.80038 |   6.76211   |
|  dynamic  | chicago |   nyc   |       5天       |  7.13852  | 6.68604 |   7.10314   |
|  static   | chicago |   nyc   |       5天       |  7.13852  | 6.68604 |   7.10062   |
|  dynamic  | chicago |   nyc   |       7天       |  7.12167  | 6.67103 |   7.09394   |
|  static   | chicago |   nyc   |       7天       |  7.12167  | 6.67103 |   7.10788   |
|  dynamic  |   dc    |   nyc   |       1天       |  9.41977  | 7.78120 |   7.75578   |
|  static   |   dc    |   nyc   |       1天       |  9.42764  | 7.78120 |   7.75579   |
|  dynamic  |   dc    |   nyc   |       3天       |  7.92233  | 6.93770 |   8.94962   |
|  static   |   dc    |   nyc   |       3天       |  7.92233  | 6.93770 |   8.86747   |
|  dynamic  |   dc    |   nyc   |       5天       |  7.87232  | 7.67169 |   7.87556   |
|  static   |   dc    |   nyc   |       5天       |  7.87232  | 7.67169 |   7.87427   |
|  dynamic  |   dc    |   nyc   |       7天       | 24.38127  | 7.73380 |   7.81032   |
|  static   |   dc    |   nyc   |       7天       |  7.81039  | 7.73380 |   7.81107   |
|  dynamic  |   dc    | chicago |       1天       | 10.58203  | 4.67799 |   4.75832   |
|  static   |   dc    | chicago |       1天       | 10.58203  | 4.67799 |   4.75832   |
|  dynamic  |   dc    | chicago |       3天       | 11.23320  | 3.82442 |   4.21488   |
|  static   |   dc    | chicago |       3天       | 11.23320  | 3.82442 |   4.00140   |
|  dynamic  |   dc    | chicago |       5天       |  9.93056  | 3.79295 |   3.91066   |
|  static   |   dc    | chicago |       5天       |  9.93056  | 3.79295 |   3.73602   |
|  dynamic  |   dc    | chicago |       7天       |  8.54751  | 4.08669 |   3.75169   |
|  static   |   dc    | chicago |       7天       |  8.54751  | 4.08669 |   3.84819   |

dynamic 和 static 训练方式的结果非常相近

