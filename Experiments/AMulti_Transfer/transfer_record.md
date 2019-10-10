
|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Distance|checkin|nyc|chicago|0.1|1天|4.70441|4.70361|12.59841|
|Distance|checkin|nyc|chicago|0.1|3天|4.70441|4.82347|4.78770|
|Distance|checkin|nyc|chicago|0.1|5天|4.70441|4.83354|4.81036|
|Distance|checkin|nyc|chicago|0.1|7天|4.70441|4.70070|4.74429|
|Distance|checkin|nyc|dc|0.1|1天|4.18998|4.18815|4.18669|
|Distance|checkin|nyc|dc|0.1|3天|4.18998|3.95583|3.95511|
|Distance|checkin|nyc|dc|0.1|5天|4.18998|3.96894|4.01603|
|Distance|checkin|chicago|dc|0.1|1天|3.19900|3.10418|3.11494|
|Distance|checkin|chicago|dc|0.1|3天|3.19900|3.00662|3.13352|
|Distance|checkin|chicago|dc|0.1|5天|3.19900|2.98474|2.99215|
|Distance|checkin|chicago|dc|0.1|7天|3.19900|2.89581|2.94413|
|Distance|checkin|chicago|nyc|0.1|1天|10.24455|9.44282|12.85142|
|Distance|checkin|chicago|nyc|0.1|3天|10.24455|6.87987|12.93270|
|Distance|checkin|chicago|nyc|0.1|5天|10.24455|10.21737|13.56898|
|Distance|checkin|chicago|nyc|0.1|7天|10.24455|6.66069|12.12317|
|Distance|checkin|dc|nyc|0.1|1天|7.23524|7.23436|7.23350|
|Distance|checkin|dc|nyc|0.1|3天|7.23524|6.73650|6.81929|
|Distance|checkin|dc|nyc|0.1|5天|7.23524|7.23440|7.23744|
|Distance|checkin|dc|nyc|0.1|7天|7.23524|7.23836|7.23745|
|Distance|checkin|dc|chicago|0.1|1天|4.34642|4.35396|4.34736|
|Distance|checkin|dc|chicago|0.1|5天|4.34642|4.49737|4.39714|
|Distance|checkin|dc|chicago|0.1|7天|4.34642|4.49232|4.37209|
|Distance|checkin|dc|chicago|0.1|3天|4.34642|4.46849|4.47618|
|Distance|checkin|nyc|chicago|0.1|1天|4.70441|4.70361|12.59841|
|Distance|checkin|dc|chicago|0.1|1天|4.34642|4.35396|4.34736|
|Distance|checkin|chicago|dc|0.1|1天|3.19900|3.10418|3.11494|

#### Tuning the transfer-ratio

|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Correlation|checkin|nyc|chicago|0.2|1天|4.67712|3.92269|3.99346|
|Correlation|checkin|nyc|chicago|0.2|3天|4.22121|3.70951|3.71976|
|Correlation|checkin|nyc|chicago|0.2|5天|4.32650|4.02932|4.06536|
|Correlation|checkin|nyc|chicago|0.2|7天|3.77726|3.61502|3.64801|
|Correlation|checkin|nyc|chicago|0.3|1天|4.67712|3.92269|3.99815|
|Correlation|checkin|nyc|chicago|0.3|3天|4.22121|3.70951|3.72503|
|Correlation|checkin|nyc|chicago|0.3|5天|4.32650|4.02932|4.03297|
|Correlation|checkin|nyc|chicago|0.3|7天|3.77726|3.61502|3.64506|
|Correlation|checkin|nyc|dc|0.2|1天|3.77775|3.41215|3.40791|
|Correlation|checkin|nyc|dc|0.2|3天|3.53591|3.25796|3.23612|
|Correlation|checkin|nyc|dc|0.2|5天|3.39463|3.39892|3.38755|
|Correlation|checkin|nyc|dc|0.2|7天|3.38310|3.29268|3.28780|
|Correlation|checkin|nyc|dc|0.3|1天|3.77775|3.41215|3.41827|
|Correlation|checkin|nyc|dc|0.3|3天|3.53591|3.25796|3.23804|
|Correlation|checkin|nyc|dc|0.3|5天|3.39463|3.39892|3.39739|
|Correlation|checkin|nyc|dc|0.3|7天|3.38310|3.29268|3.29109|
|Correlation|checkin|chicago|dc|0.2|1天|3.78157|3.71789|3.69212|
|Correlation|checkin|chicago|dc|0.2|3天|3.69740|3.14304|3.24586|
|Correlation|checkin|chicago|dc|0.2|5天|3.62814|3.17226|3.19682|
|Correlation|checkin|chicago|dc|0.2|7天|3.58433|3.11757|3.16523|
|Correlation|checkin|chicago|dc|0.3|1天|3.78157|3.71789|3.69082|
|Correlation|checkin|chicago|dc|0.3|3天|3.69740|3.14304|3.18183|
|Correlation|checkin|chicago|dc|0.3|5天|3.62814|3.17226|3.20910|
|Correlation|checkin|chicago|dc|0.3|7天|3.58433|3.11757|3.18558|
|Correlation|checkin|chicago|nyc|0.2|1天|7.54476|6.90001|7.05806|
|Correlation|checkin|chicago|nyc|0.2|3天|7.22262|6.81500|6.82105|
|Correlation|checkin|chicago|nyc|0.2|5天|7.13852|6.79075|7.10494|
|Correlation|checkin|chicago|nyc|0.2|7天|7.12167|6.56041|7.08145|
|Correlation|checkin|chicago|nyc|0.3|1天|7.33929|7.14455|7.04937|
|Correlation|checkin|chicago|nyc|0.3|3天|7.22262|6.86183|7.53325|
|Correlation|checkin|chicago|nyc|0.3|5天|7.13852|6.71483|7.11741|
|Correlation|checkin|chicago|nyc|0.3|7天|7.12167|6.56041|7.10332|
|Correlation|checkin|dc|nyc|0.2|1天|9.42764|7.78120|7.59939|
|Correlation|checkin|dc|nyc|0.2|3天|7.92233|6.85930|7.58509|
|Correlation|checkin|dc|nyc|0.2|5天|20.37170|7.60804|7.87388|
|Correlation|checkin|dc|nyc|0.2|7天|7.81039|6.86581|7.75423|
|Correlation|checkin|dc|nyc|0.3|1天|9.42764|7.78120|7.65509|
|Correlation|checkin|dc|nyc|0.3|3天|7.92233|6.85930|7.49069|
|Correlation|checkin|dc|nyc|0.3|5天|7.87232|7.60804|16.16995|
|Correlation|checkin|dc|nyc|0.3|7天|7.81039|6.86581|7.81060|
|Correlation|checkin|dc|chicago|0.2|1天|10.58203|4.67799|4.73050|
|Correlation|checkin|dc|chicago|0.2|3天|11.23320|3.90123|3.57875|
|Correlation|checkin|dc|chicago|0.2|5天|9.93056|3.79625|3.40253|
|Correlation|checkin|dc|chicago|0.2|7天|8.54751|4.06583|3.40602|
|Correlation|checkin|dc|chicago|0.3|1天|10.58203|4.67799|4.88579|
|Correlation|checkin|dc|chicago|0.3|3天|11.23320|3.90123|3.59715|
|Correlation|checkin|dc|chicago|0.3|5天|9.93056|3.79625|3.43136|
|Correlation|checkin|dc|chicago|0.3|7天|8.54751|4.06583|3.42706|
|Correlation|checkin|nyc|chicago|0.05|1天|4.67712|3.92269|3.97959|
|Correlation|checkin|nyc|chicago|0.05|3天|4.22121|3.70951|3.73577|
|Correlation|checkin|nyc|chicago|0.05|5天|4.32650|4.02932|4.10112|
|Correlation|checkin|nyc|chicago|0.05|7天|3.77726|3.61502|3.63231|
|Correlation|checkin|nyc|dc|0.05|1天|3.77775|3.41215|3.24400|
|Correlation|checkin|nyc|dc|0.05|3天|3.53591|3.25796|3.24783|
|Correlation|checkin|nyc|dc|0.05|5天|3.39463|3.39892|3.38714|
|Correlation|checkin|nyc|dc|0.05|7天|3.38310|3.29268|3.27108|
|Correlation|checkin|chicago|dc|0.05|1天|3.78157|3.71789|3.69419|
|Correlation|checkin|chicago|dc|0.05|3天|3.69740|3.14304|3.14949|
|Correlation|checkin|chicago|dc|0.05|5天|3.62814|3.17226|3.17349|
|Correlation|checkin|chicago|dc|0.05|7天|3.58433|3.11757|3.12788|
|Correlation|checkin|chicago|nyc|0.05|1天|7.33929|6.90001|6.94973|
|Correlation|checkin|chicago|nyc|0.05|3天|7.22262|7.01002|6.79002|
|Correlation|checkin|chicago|nyc|0.05|5天|7.13852|6.71483|7.11115|
|Correlation|checkin|chicago|nyc|0.05|7天|7.12167|6.56041|7.09967|
|Correlation|checkin|dc|nyc|0.05|1天|9.42764|7.78120|7.38421|
|Correlation|checkin|dc|nyc|0.05|3天|7.92233|6.85930|12.42980|
|Correlation|checkin|dc|nyc|0.05|5天|7.87232|7.60804|7.80362|
|Correlation|checkin|dc|nyc|0.05|7天|7.81039|6.86581|7.81139|
|Correlation|checkin|dc|chicago|0.05|1天|10.58203|4.67799|4.54821|
|Correlation|checkin|dc|chicago|0.05|3天|11.23320|3.90123|3.58257|
|Correlation|checkin|dc|chicago|0.05|5天|9.93056|3.79625|3.43679|
|Correlation|checkin|dc|chicago|0.05|7天|8.54751|4.06583|3.39452|

#### Dynamic mode

| TrainMode |   SD    |   TD    | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :-----: | :-----: | :------------: | :-------: | :-----: | :---------: | :---------: |
|dynamic|nyc|chicago|1天|4.67712|3.92269|3.99061|
|dynamic|nyc|chicago|3天|4.22121|3.72731|3.72169|
|dynamic|nyc|chicago|5天|4.32650|4.03527|4.09383|
|dynamic|nyc|chicago|7天|3.77726|3.62886|3.64174|
|dynamic|nyc|dc|1天|3.77775|3.41219|3.24365|
|dynamic|nyc|dc|3天|3.53591|3.24496|3.23717|
|dynamic|nyc|dc|5天|3.39463|3.39877|3.41054|
|dynamic|nyc|dc|7天|3.38310|3.29157|3.27758|
|dynamic|chicago|dc|1天|3.78157|3.71789|3.69317|
|dynamic|chicago|dc|3天|3.69740|3.14631|3.18608|
|dynamic|chicago|dc|5天|3.62814|3.17078|3.19249|
|dynamic|chicago|dc|7天|3.58433|3.11549|3.15131|
|dynamic|chicago|nyc|1天|7.33929|7.38597|6.96449|
|dynamic|chicago|nyc|3天|7.55972|6.80038|6.77189|
|dynamic|chicago|nyc|5天|7.13852|6.68604|7.10314|
|dynamic|chicago|nyc|7天|7.12167|6.67103|7.09394|
|dynamic|dc|nyc|1天|9.41977|7.78120|25.48727|
|dynamic|dc|nyc|3天|7.92233|6.93770|8.94962|
|dynamic|dc|nyc|5天|7.87232|7.67169|7.87556|
|dynamic|dc|nyc|7天|24.38127|7.73380|7.81032|
|dynamic|dc|chicago|1天|10.58203|4.67799|4.75832|
|dynamic|dc|chicago|3天|11.23320|3.82442|4.21488|
|dynamic|dc|chicago|5天|9.93056|3.79295|3.91066|
|dynamic|dc|chicago|7天|8.54751|4.08669|3.75169||Correlation|checkin|dc|nyc|0.1|1天|9.42764|7.78120|14.34273|
|Correlation|checkin|dc|nyc|0.1|1天|9.42764|7.78120|7.75579|



|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Correlation|checkin|nyc|chicago|0.1|1天|4.76519|3.91774|4.00033|
|Correlation|checkin|nyc|chicago|0.1|3天|4.21627|3.69943|3.71308|
|Correlation|checkin|nyc|chicago|0.1|7天|3.77277|3.62399|3.64341|
|Correlation|checkin|nyc|dc|0.1|1天|3.77953|3.40973|3.27021|
|Correlation|checkin|nyc|dc|0.1|3天|3.53572|3.25625|3.24498|
|Correlation|checkin|nyc|dc|0.1|5天|3.39470|3.39883|3.39262|
|Correlation|checkin|nyc|dc|0.1|7天|3.38312|3.29258|3.27573|
|Correlation|checkin|chicago|dc|0.1|1天|3.78083|3.71714|3.69250|
|Correlation|checkin|chicago|dc|0.1|3天|3.69748|3.14352|3.15632|
|Correlation|checkin|chicago|dc|0.1|5天|3.62816|3.17147|3.17755|
|Correlation|checkin|chicago|dc|0.1|7天|3.58449|3.11533|3.14646|
|Correlation|checkin|chicago|nyc|0.1|1天|7.33863|6.90045|6.96378|
|Correlation|checkin|chicago|nyc|0.1|3天|7.22249|6.85352|6.79605|
|Correlation|checkin|chicago|nyc|0.1|5天|7.13851|6.70580|7.09479|
|Correlation|checkin|chicago|nyc|0.1|7天|7.12164|6.69556|7.10098|
|Correlation|checkin|dc|nyc|0.1|1天|9.38367|7.77615|7.49206|
|Correlation|checkin|dc|nyc|0.1|3天|7.92180|6.91332|7.66319|
|Correlation|checkin|dc|nyc|0.1|5天|7.87216|7.61887|7.87342|
|Correlation|checkin|dc|nyc|0.1|7天|7.81035|7.09896|7.81083|
|Correlation|checkin|dc|chicago|0.1|1天|10.60726|4.71315|4.52085|
|Correlation|checkin|nyc|chicago|0.1|1天|4.40424|6.09647|4.68890|
|Correlation|checkin|nyc|chicago|0.1|1天|4.40412|6.10487|4.69186|
