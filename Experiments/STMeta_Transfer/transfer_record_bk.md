#### Correlation graph transfer result, match using traffic flow (30 days)

|   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: |
|nyc|chicago|0.1|1天|4.67712|4.67061|**4.50287**|
|nyc|chicago|0.1|3天|4.22121|**3.86823**|4.97310|
|nyc|chicago|0.1|5天|4.32650|**3.83584**|3.89576|
|nyc|chicago|0.1|7天|3.77726|**3.64761**|3.69924|
|nyc|dc|0.1|1天|3.77775|3.60307|**3.31033**|
|nyc|dc|0.1|3天|3.53591|**3.19822**|3.27654|
|nyc|dc|0.1|5天|3.39463|3.31562|**3.31375**|
|nyc|dc|0.1|7天|3.38310|3.22907|**3.22316**|
|dc|nyc|0.1|1天|9.42764|**7.69555**|8.03807|
|dc|nyc|0.1|3天|7.92233|**7.86374**|7.87504|
|dc|nyc|0.1|5天|7.87232|**7.57717**|7.87731|
|dc|nyc|0.1|7天|7.81039|**7.74953**|7.79245|
|dc|chicago|0.1|1天|10.58203|5.11785|**4.89390**|
|dc|chicago|0.1|3天|11.23320|12.65670|**4.74284**|
|dc|chicago|0.1|5天|9.93056|4.28090|**3.47838**|
|dc|chicago|0.1|7天|8.54751|**3.32681**|3.38316|
|chicago|nyc|0.1|1天|7.33929|**6.98808**|7.01758|
|chicago|nyc|0.1|3天|7.22262|**7.03898**|7.28277|
|chicago|nyc|0.1|5天|7.13852|**6.80380**|7.12489|
|chicago|nyc|0.1|7天|7.12167|**6.63963**|7.07976|
|chicago|dc|0.1|1天|3.78157|3.75912|**3.75336**|
|chicago|dc|0.1|3天|3.69740|**3.58552**|3.62116|
|chicago|dc|0.1|5天|3.62814|**3.15233**|3.40723|

#### Correlation graph transfer result，match with check-in feature

|   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: |
|nyc|chicago|0.1|1天|4.67712|4.67061|**4.49448**|
|nyc|chicago|0.1|3天|4.22121|4.52615|**3.87400**|
|nyc|chicago|0.1|5天|4.32650|**3.89086**|3.94352|
|nyc|chicago|0.1|7天|3.77726|**3.64852**|3.69911|
|nyc|dc|0.1|1天|3.77775|3.60307|**3.27499**|
|nyc|dc|0.1|3天|3.53591|3.42623|**3.38511**|
|nyc|dc|0.1|5天|3.39463|**3.29540**|3.32153|
|nyc|dc|0.1|7天|3.38310|3.24432|**3.23480**|
|dc|nyc|0.1|1天|9.42764|**7.69555**|7.95242|
|dc|nyc|0.1|3天|7.92233|**7.87700**|8.12669|
|dc|nyc|0.1|5天|7.87232|**7.64216**|7.86986|
|dc|nyc|0.1|7天|7.81039|**7.74056**|7.79816|
|dc|chicago|0.1|1天|10.58203|5.11785|**4.88025**|
|dc|chicago|0.1|3天|11.23320|5.46448|**4.44395**|
|dc|chicago|0.1|5天|9.93056|4.30427|**3.44833**|
|dc|chicago|0.1|7天|8.54751|**3.28179**|3.34347|
|chicago|nyc|0.1|1天|7.33929|**6.98808**|7.02008|
|chicago|nyc|0.1|3天|7.22262|7.36530|**7.00954**|
|chicago|nyc|0.1|5天|7.13852|**6.63782**|7.13790|
|chicago|nyc|0.1|7天|7.12167|**6.73726**|7.09331|
|chicago|dc|0.1|1天|3.78157|3.75912|**3.75527**|
|chicago|dc|0.1|3天|3.69740|**3.61495**|3.62228|
|chicago|dc|0.1|5天|3.62814|**3.14687**|3.17531|
|chicago|dc|0.1|7天|3.58433|**3.07332**|3.10778|

#### Distance Graph，Check-In feature，Dynamic training

|   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: |
|nyc|chicago|0.1|1天|**4.70441**|5.17771|5.13792|
|nyc|chicago|0.1|3天|**4.70441**|5.24400|5.40755|
|nyc|chicago|0.1|5天|4.70441|4.71439|**4.68781**|
|nyc|chicago|0.1|7天|4.70441|4.71179|**4.57209**|
|nyc|dc|0.1|1天|4.18998|4.18247|**3.94269**|
|nyc|dc|0.1|3天|4.18998|**4.13482**|4.14903|
|nyc|dc|0.1|5天|4.18998|3.86709|**3.86132**|
|nyc|dc|0.1|7天|4.18998|3.63065|**3.61484**|
|dc|nyc|0.1|1天|7.23524|**7.23120**|8.61836|
|dc|nyc|0.1|3天|7.23524|**6.33199**|6.35476|
|dc|nyc|0.1|5天|**7.23524**|7.24360|8.84953|
|dc|nyc|0.1|7天|**7.23524**|7.26807|9.24907|
|dc|chicago|0.1|1天|4.34642|4.37925|**4.35733**|
|dc|chicago|0.1|3天|**4.34642**|4.39791|4.54773|
|dc|chicago|0.1|5天|**4.34642**|4.51153|4.46805|
|dc|chicago|0.1|7天|4.34642|4.04295|**4.00917**|
|chicago|nyc|0.1|1天|**10.24455**|12.02121|15.36785|
|chicago|nyc|0.1|3天|**10.24455**|12.11566|15.38250|
|chicago|nyc|0.1|5天|**10.24455**|10.90358|13.41528|
|chicago|nyc|0.1|7天|10.24455|6.71849|**6.71550**|
|chicago|dc|0.1|1天|3.19900|**3.09946**|3.23995|
|chicago|dc|0.1|3天|3.19900|3.14234|**3.12200**|
|chicago|dc|0.1|5天|3.19900|**2.99568**|3.16543|
|chicago|dc|0.1|7天|3.19900|**2.92313**|3.07769|

#### Distance Graph，Check-In feature，Dynamic training

|   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: |
|nyc|chicago|0.1|1天|**4.70441**|5.17771|5.13792|
|nyc|chicago|0.1|3天|**4.70441**|5.02598|4.85231|
|nyc|chicago|0.1|5天|4.70441|4.79848|**4.65276**|
|nyc|chicago|0.1|7天|4.70441|4.72065|**4.58330**|
|nyc|dc|0.1|1天|4.18998|4.18247|**3.94269**|
|nyc|dc|0.1|3天|4.18998|4.02759|**3.91673**|
|nyc|dc|0.1|5天|4.18998|3.87307|**3.85373**|
|nyc|dc|0.1|7天|4.18998|3.66314|**3.61744**|
|dc|nyc|0.1|1天|7.23524|**7.23120**|8.61836|
|dc|nyc|0.1|3天|7.23524|7.18937|**6.76861**|
|dc|nyc|0.1|5天|**7.23524**|7.24310|8.85444|
|dc|nyc|0.1|7天|**7.23524**|7.26563|9.23842|
|dc|chicago|0.1|1天|**4.34642**|4.37925|4.35733|
|dc|chicago|0.1|3天|**4.34642**|4.39091|4.61458|
|dc|chicago|0.1|5天|**4.34642**|4.51621|4.47188|
|dc|chicago|0.1|7天|4.34642|4.07226|**4.03542**|
|chicago|nyc|0.1|1天|**10.24455**|12.02121|15.36785|
|chicago|nyc|0.1|3天|**10.24455**|10.99130|16.56082|
|chicago|nyc|0.1|5天|10.24455|**6.69829**|6.78793|
|chicago|nyc|0.1|7天|10.24455|**6.89889**|7.57003|
|chicago|dc|0.1|1天|3.19900|**3.09946**|3.23995|
|chicago|dc|0.1|3天|3.19900|**3.12949**|3.18817|
|chicago|dc|0.1|5天|3.19900|**3.02039**|3.18615|
|chicago|dc|0.1|7天|3.19900|**2.90374**|3.11075|

|   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: |
|nyc|chicago|0.1|1天|4.70441|5.17771|5.13792|
|nyc|chicago|0.1|3天|4.70441|4.80959|4.94123|
|nyc|chicago|0.1|5天|4.70441|4.77160|4.68924|
|nyc|chicago|0.1|7天|4.70441|4.77123|4.57228|
|nyc|dc|0.1|1天|4.18998|4.18247|3.94269|
|nyc|dc|0.1|3天|4.18998|4.05417|4.09441|
|nyc|dc|0.1|5天|4.18998|3.92515|3.92526|
|nyc|dc|0.1|7天|4.18998|3.67485|3.62186|
|dc|nyc|0.1|1天|7.23524|7.23120|8.61836|
|dc|nyc|0.1|3天|7.23524|6.36652|6.29304|
|dc|nyc|0.1|5天|7.23524|8.12606|8.13084|
|dc|nyc|0.1|7天|7.23524|7.26382|9.24113|
|dc|chicago|0.1|1天|4.34642|4.37925|4.35733|
|dc|chicago|0.1|3天|4.34642|4.40805|4.60000|
|dc|chicago|0.1|5天|4.34642|4.49298|4.48141|
|dc|chicago|0.1|7天|4.34642|4.03225|3.99625|
|chicago|nyc|0.1|1天|10.24455|12.02121|15.36785|
|chicago|nyc|0.1|3天|10.24455|12.84052|12.39428|
|chicago|nyc|0.1|5天|10.24455|10.32351|13.51593|
|chicago|nyc|0.1|7天|10.24455|11.71842|11.97301|
|chicago|dc|0.1|1天|3.19900|3.09946|3.23995|
|chicago|dc|0.1|3天|3.19900|3.09866|3.20144|
|chicago|dc|0.1|5天|3.19900|3.03267|3.18821|
|chicago|dc|0.1|7天|3.19900|2.90296|3.09439|

#### Running tests

|   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: |
|nyc|chicago|0.1|1天|4.67712|4.17206|4.35158|
|nyc|chicago|0.1|3天|4.22121|4.40948|4.67440|
|nyc|chicago|0.1|5天|4.32650|3.81700|3.92837|
|nyc|chicago|0.1|7天|3.77726|3.63045|3.66386|
|nyc|dc|0.1|1天|3.77775|3.23772|3.47581|
|nyc|dc|0.1|3天|3.53591|3.30477|3.81137|
|nyc|dc|0.1|5天|3.39463|3.32380|3.32734|
|nyc|dc|0.1|7天|3.38310|3.24348|3.26136|

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
||nyc|chicago|0.3|1天|4.67712|4.17237|4.04008|
||nyc|chicago|0.3|3天|4.22121|4.22041|4.08511|
||nyc|chicago|0.3|5天|4.32650|3.94659|3.86648|
||nyc|chicago|0.3|7天|3.77726|3.64293|3.65762|
||nyc|dc|0.3|1天|3.77775|3.23772|3.44210|
||nyc|dc|0.3|3天|3.53591|3.36918|3.32675|
||nyc|dc|0.3|5天|3.39463|3.32176|3.32761|
||nyc|dc|0.3|7天|3.38310|3.24480|3.34377|
||nyc|dc|0.5|1天|3.77775|3.40004|3.78089|
||nyc|dc|0.3|1天|3.77775|3.39991|3.52524|
||nyc|dc|0.3|3天|3.53591|3.23931|3.46197|
||nyc|dc|0.3|5天|3.39463|3.31526|3.35127|
||nyc|dc|0.3|7天|3.38310|3.23292|3.26009|
||nyc|dc|0.2|1天|3.77775|3.39991|3.48174|
||nyc|dc|0.2|3天|3.53591|3.23931|3.36759|
||nyc|dc|0.2|5天|3.39463|3.31526|3.36462|
||nyc|dc|0.2|7天|3.38310|3.23292|3.31145|
||nyc|dc|0.1|1天|3.77775|3.39991|3.28871|
||nyc|dc|0.1|3天|3.53591|3.23931|3.37452|
||nyc|dc|0.1|5天|3.39463|3.31526|3.33550|
||nyc|dc|0.1|7天|3.38310|3.23292|3.30978|
||nyc|dc|0.05|1天|3.77775|3.39991|3.33861|
||nyc|dc|0.05|3天|3.53591|3.23931|3.41795|
||nyc|dc|0.05|5天|3.39463|3.31526|3.32382|
||nyc|dc|0.05|7天|3.38310|3.23292|3.23323|
||nyc|dc|0.1|1天|3.77775|3.39991|3.41954|
||nyc|dc|0.1|3天|3.53591|3.23931|3.25412|
||nyc|dc|0.1|5天|3.39463|3.31526|3.25395|
||nyc|dc|0.1|7天|3.38310|3.23292|3.24065|
||nyc|dc|0.2|1天|3.77775|3.39991|3.41859|
||nyc|dc|0.2|3天|3.53591|3.23931|3.24894|
||nyc|dc|0.2|5天|3.39463|3.31526|3.23863|
||nyc|dc|0.2|7天|3.38310|3.23292|3.24001|

1. Correlation Graph 比 Distance Graph 更容易迁移
2. 目标区域的数据质量可能会影响迁移效果
3. transfer ratio可以相应做出调整，会影响迁移的效果

新的实验：

1. 在表格中加入graph类型
2. 调整测试集长度、以调整目标区域训练数据到工作日区间

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|C|nyc|chicago|0.3|1天|4.20614|3.95398|4.17578|
|C|nyc|chicago|0.3|3天|3.96698|4.31441|4.26729|
|C|nyc|chicago|0.3|5天|4.02681|3.58041|3.64260|
|C|nyc|chicago|0.3|3天|3.96698|3.96458|3.98778|
|nyc|dc|0.1|1天|3.63295|3.23437|4.71338|
|nyc|dc|0.1|1天|3.63295|3.23437|4.70039|
|nyc|dc|0.1|1天|3.63295|3.23437|4.70039|
|nyc|dc|0.1|1天|3.63295|3.23437|3.90929|
|C|nyc|dc|0.1|1天|3.63295|3.51374|3.53088|

## Static Training Result

1. val method: use 20% from training data
2. test data length: about 30 days

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|Correlation|nyc|chicago|0.1|1天|4.67712|**4.25955**|4.49069|
|Correlation|nyc|chicago|0.1|3天|4.22121|3.85813|**3.68496**|
|Correlation|nyc|chicago|0.1|5天|4.32650|**3.96839**|4.02760|
|Correlation|nyc|chicago|0.1|7天|3.77726|**3.63063**|3.64774|
|Correlation|nyc|dc|0.1|1天|3.77775|**3.23351**|3.30394|
|Correlation|nyc|dc|0.1|3天|3.53591|**3.20542**|3.30849|
|Correlation|nyc|dc|0.1|5天|3.39463|3.36513|**3.35833**|
|Correlation|nyc|dc|0.1|7天|3.38310|**3.27965**|3.30015|
|Correlation|dc|nyc|0.1|1天|9.42764|7.56931|**7.30441**|
|Correlation|dc|nyc|0.1|3天|7.92233|**7.85794**|7.87110|
|Correlation|dc|nyc|0.1|5天|7.87232|**7.78041**|7.86299|
|Correlation|dc|nyc|0.1|7天|7.81039|~~11.09254~~|**7.80470**|
|Correlation|dc|chicago|0.1|1天|10.58203|4.98520|**4.73672**|
|Correlation|dc|chicago|0.1|3天|11.23320|4.55043|**4.06235**|
|Correlation|dc|chicago|0.1|5天|9.93056|3.65179|**3.52144**|
|Correlation|dc|chicago|0.1|7天|8.54751|4.09174|**3.94603**|
|Correlation|chicago|nyc|0.1|1天|7.33929|7.12667|**7.10164**|
|Correlation|chicago|nyc|0.1|3天|7.22262|6.92887|**6.81944**|
|Correlation|chicago|nyc|0.1|5天|7.13852|**6.69150**|7.11825|
|Correlation|chicago|nyc|0.1|7天|7.12167|**6.71981**|7.13281|
|Correlation|chicago|dc|0.1|1天|3.78157|3.77235|**3.75456**|
|Correlation|chicago|dc|0.1|3天|3.69740|**3.40131**|3.53313|
|Correlation|chicago|dc|0.1|5天|3.62814|**3.13873**|3.16228|
|Correlation|chicago|dc|0.1|7天|3.58433|**3.09991**|3.14930|

Problem : 部分的Fine-tune结果，训练数据增加后，效果变差；迁移的效果不稳定；

进行以下修改：

1. reduce learning rate, to stable the training process，增加batch size
2. 目前目前区域的训练数据大约在周三，以下调整到周五，这样保证大部分数据在工作日

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|Correlation|nyc|chicago|0.3|1天|5.38677|4.94947|5.30289|
|Correlation|nyc|chicago|0.3|3天|4.34201|3.86219|3.98421|
|Correlation|nyc|chicago|0.3|5天|4.06206|3.82933|3.85310|
|Correlation|nyc|chicago|0.3|7天|3.72066|3.60633|3.61164|
|Correlation|nyc|dc|0.3|1天|4.38434|4.87484|4.98700|
|Correlation|nyc|dc|0.3|3天|3.41225|3.57403|3.59285|
|Correlation|nyc|dc|0.3|5天|3.44476|3.35405|3.30752|
|Correlation|nyc|dc|0.3|7天|3.39200|3.37872|3.28680|
|Correlation|dc|nyc|0.3|1天|8.34917|8.34650|18.33073|
|Correlation|dc|nyc|0.3|3天|7.94347|7.88922|7.93095|
|Correlation|dc|nyc|0.3|5天|7.90373|7.89213|7.90797|
|Correlation|dc|nyc|0.3|7天|7.90058|7.87062|7.90461|
|Correlation|dc|chicago|0.3|1天|10.86334|4.99778|5.94911|
|Correlation|dc|chicago|0.3|3天|11.46182|5.25774|4.26376|
|Correlation|dc|chicago|0.3|5天|10.17678|4.09519|3.56348|
|Correlation|dc|chicago|0.3|7天|8.70574|5.25094|3.40993|
|Correlation|chicago|nyc|0.3|1天|7.36875|7.47049|7.47635|
|Correlation|chicago|nyc|0.3|3天|7.26527|7.25193|7.22621|
|Correlation|chicago|nyc|0.3|5天|7.25791|7.23149|7.22586|


| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|Correlation|nyc|chicago|0.1|1天|5.38677|4.94947|5.30022|
|Correlation|nyc|chicago|0.1|3天|4.34201|3.94191|4.13253|
|Correlation|nyc|chicago|0.1|5天|4.06206|3.82461|3.94119|
|Correlation|nyc|chicago|0.1|7天|3.72066|3.60703|3.60470|
|Correlation|nyc|dc|0.1|1天|4.38434|4.87484|4.98315|
|Correlation|nyc|dc|0.1|3天|3.41225|3.78044|3.42910|
|Correlation|nyc|dc|0.1|5天|3.44476|3.33215|3.31266|
|Correlation|nyc|dc|0.1|7天|3.39200|3.37359|3.26246|
|Correlation|dc|nyc|0.1|1天|8.34917|8.34650|8.32973|
|Correlation|dc|nyc|0.1|3天|7.94347|8.05051|7.94830|
|Correlation|dc|nyc|0.1|5天|7.90373|7.90132|7.90560|
|Correlation|dc|nyc|0.1|7天|7.90058|7.87989|7.90426|
|Correlation|dc|chicago|0.1|1天|10.86334|4.99787|5.76370|
|Correlation|dc|chicago|0.1|3天|11.46182|5.91710|4.59539|
|Correlation|dc|chicago|0.1|5天|10.17678|4.44333|3.61311|
|Correlation|dc|chicago|0.1|7天|8.70574|3.63362|3.45863|
|Correlation|chicago|nyc|0.1|1天|7.36875|7.47049|7.45168|
|Correlation|chicago|nyc|0.1|3天|7.26527|7.25407|8.32680|
|Correlation|chicago|nyc|0.1|5天|7.25791|7.23103|7.25204|
|Correlation|chicago|nyc|0.1|7天|7.24833|7.20168|7.23582|
|Correlation|chicago|dc|0.1|1天|3.86311|4.60512|3.91758|
|Correlation|chicago|dc|0.1|3天|3.66394|3.32028|3.61588|
|Correlation|chicago|dc|0.1|5天|3.62543|3.52207|3.56268|
|Correlation|chicago|dc|0.1|7天|3.59582|3.56239|3.53162|

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|Correlation|nyc|chicago|0.05|1天|5.38677|4.94947|5.30021|
|Correlation|nyc|chicago|0.05|3天|4.34201|4.21479|3.94523|
|Correlation|nyc|chicago|0.05|5天|4.06206|3.84668|3.91428|
|Correlation|nyc|chicago|0.05|7天|3.72066|3.62060|3.60362|
|Correlation|nyc|dc|0.05|1天|4.38434|4.87484|4.98234|
|Correlation|nyc|dc|0.05|3天|3.41225|3.51476|3.68511|
|Correlation|nyc|dc|0.05|5天|3.44476|3.28846|3.34820|
|Correlation|nyc|dc|0.05|7天|3.39200|3.37931|3.27885|
|Correlation|dc|nyc|0.05|1天|8.34917|8.34650|8.33439|
|Correlation|dc|nyc|0.05|3天|7.94347|7.58449|7.93117|
|Correlation|dc|nyc|0.05|5天|7.90373|7.89797|7.90328|
|Correlation|dc|nyc|0.05|7天|7.90058|7.89736|7.90086|
|Correlation|dc|chicago|0.05|1天|10.86334|4.99778|5.82246|
|Correlation|dc|chicago|0.05|3天|11.46182|5.09725|4.57277|
|Correlation|dc|chicago|0.05|5天|10.17678|4.13495|3.85751|
|Correlation|dc|chicago|0.05|7天|8.70574|3.61414|3.76792|
|Correlation|chicago|nyc|0.05|1天|7.36875|7.47049|7.45205|
|Correlation|chicago|nyc|0.05|3天|7.26527|7.25339|8.75825|
|Correlation|chicago|nyc|0.05|5天|7.25791|7.23152|7.24569|
|Correlation|chicago|nyc|0.05|7天|7.24833|7.19456|7.24255|
|Correlation|chicago|dc|0.05|1天|3.86311|4.60512|3.89865|
|Correlation|chicago|dc|0.05|3天|3.66394|3.48957|3.45902|
|Correlation|chicago|dc|0.05|5天|3.62543|3.53435|3.55831|
|Correlation|chicago|dc|0.05|7天|3.59582|3.50022|3.52657|

Change Back to 周三 setting， and test other transfer ratio，change lr back to 0.0005

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|Correlation|nyc|chicago|0.2|1天|4.67712|4.26000|4.37578|
|Correlation|nyc|chicago|0.2|3天|4.22121|3.81024|4.21875|
|Correlation|nyc|chicago|0.2|5天|4.32650|3.84399|3.93527|
|Correlation|nyc|chicago|0.2|7天|3.77726|3.62769|3.65340|
|Correlation|nyc|dc|0.2|1天|3.77775|3.23351|3.48805|
|Correlation|nyc|dc|0.2|3天|3.53591|3.37679|3.35672|
|Correlation|nyc|dc|0.2|5天|3.39463|3.28378|3.38352|
|Correlation|nyc|dc|0.2|7天|3.38310|3.26814|3.30668|
|Correlation|dc|nyc|0.2|1天|9.42764|7.56931|7.65780|
|Correlation|dc|nyc|0.2|3天|7.92233|7.56916|7.85753|
|Correlation|dc|nyc|0.2|5天|7.87232|7.80455|7.87182|
|Correlation|dc|nyc|0.2|7天|7.81039|7.56197|7.81538|
|Correlation|dc|chicago|0.2|1天|10.58203|4.98520|4.97218|
|Correlation|dc|chicago|0.2|3天|11.23320|4.18053|3.68141|
|Correlation|dc|chicago|0.2|5天|9.93056|3.62506|3.43282|
|Correlation|dc|chicago|0.2|7天|8.54751|4.09224|3.36984|
|Correlation|chicago|nyc|0.2|1天|7.33929|7.12667|7.08104|
|Correlation|chicago|nyc|0.2|3天|7.22262|7.27643|6.83507|
|Correlation|chicago|nyc|0.2|5天|7.13852|6.78991|7.12228|
|Correlation|chicago|nyc|0.2|7天|7.12167|6.69170|7.12530|
|Correlation|chicago|dc|0.2|1天|3.78157|3.77235|3.75854|
|Correlation|chicago|dc|0.2|3天|3.69740|3.26502|3.50998|
|Correlation|chicago|dc|0.2|5天|3.62814|3.13389|3.17730|
|Correlation|chicago|dc|0.2|7天|3.58433|3.11130|3.16027|

Take a smaller lr = 1e-5,

| Graph |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |    TD-FT    | TD-Transfer |
| :-----: | :-----: | :-----: | :-------------: | :-------: | :---------: | :---------: | :---------: |
|Correlation|nyc|chicago|0.2|1天|4.67712|4.25025|4.48849|
|Correlation|nyc|chicago|0.2|1天|4.67712|4.64628|4.38959|
|Correlation|nyc|chicago|0.2|3天|4.22121|4.10957|3.88816|
|Correlation|nyc|chicago|0.2|5天|4.32650|3.96755|4.01513|
|Correlation|nyc|chicago|0.2|7天|3.77726|3.61148|3.63531|
|Correlation|nyc|dc|0.2|1天|3.77775|3.41926|3.47251|
|Correlation|nyc|dc|0.2|3天|3.53591|3.47903|3.28438|
|Correlation|nyc|dc|0.2|5天|3.39463|3.32894|3.36384|
|Correlation|nyc|dc|0.2|7天|3.38310|3.25412|3.25860|
|Correlation|dc|nyc|0.2|1天|9.42764|7.24540|7.62634|
|Correlation|dc|nyc|0.2|3天|7.92233|7.88247|7.84188|
|Correlation|dc|nyc|0.2|5天|7.87232|7.75617|7.86730|
|Correlation|dc|nyc|0.2|7天|7.81039|7.25333|7.78785|
|Correlation|dc|chicago|0.2|1天|10.58203|5.07799|4.99096|
|Correlation|dc|chicago|0.2|3天|11.23320|4.29366|3.70654|
|Correlation|nyc|chicago|0.3|1天|4.67712|4.64628|4.45747|
|Correlation|nyc|chicago|0.3|1天|4.67712|3.86177|4.00894|
|Correlation|nyc|chicago|0.35|1天|4.67712|3.86177|4.01409|
|Correlation|nyc|chicago|0.35|3天|4.22121|3.73511|3.91791|
|Correlation|nyc|chicago|0.35|1天|4.67712|3.86177|4.01409|
|Correlation|nyc|chicago|0.35|3天|4.22121|3.91369|3.82390|
|Correlation|nyc|chicago|0.35|5天|4.32650|3.83038|4.13916|
|Correlation|nyc|chicago|0.35|7天|3.77726|3.64188|3.68836|
|Correlation|nyc|chicago|0.35|1天|4.67712|3.86177|4.01409|
|Correlation|nyc|chicago|0.35|3天|4.22121|3.77503|3.80009|
|Correlation|nyc|chicago|0.35|5天|4.32650|3.82202|3.90571|
|Correlation|nyc|chicago|0.35|7天|3.77726|3.63230|3.68506|
|Correlation|nyc|chicago|0.35|1天|4.67712|4.17959|4.33638|
|Correlation|nyc|chicago|0.35|1天|4.67712|3.92269|4.00094|
|Correlation|nyc|chicago|0.1|1天|4.67712|3.92269|3.98607|
|Correlation|nyc|chicago|0.1|1天|4.67712|3.92269|3.98607|
|Correlation|nyc|chicago|0.1|1天|4.67712|4.34035|4.44132|
|Correlation|nyc|chicago|0.1|1天|4.67712|3.92269|3.98607|
|Correlation|nyc|chicago|0.1|3天|4.22121|3.70477|3.79106|
|Correlation|nyc|chicago|0.1|5天|4.32650|4.02529|4.07782|
|Correlation|nyc|chicago|0.1|7天|3.77726|3.64502|3.64874|
|Correlation|nyc|chicago|0.1|1天|4.67712|3.85330|3.94753|



|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Correlation|checkin|nyc|chicago|0.1|1天|4.67712|4.36085|3.98205|
|Correlation|checkin|nyc|chicago|0.1|3天|4.22121|3.78306|3.72780|
|Correlation|checkin|nyc|chicago|0.1|5天|4.32650|4.04593|4.03227|
|Correlation|checkin|nyc|chicago|0.1|7天|3.77726|3.61688|3.61353|
|Correlation|checkin|nyc|dc|0.1|1天|3.77775|3.26938|3.27214|
|Correlation|checkin|nyc|dc|0.1|3天|3.53591|3.26024|3.23687|
|Correlation|checkin|nyc|dc|0.1|5天|3.39463|3.38980|3.38501|
|Correlation|checkin|nyc|dc|0.1|7天|3.38310|3.31886|3.30954|
|Correlation|checkin|chicago|dc|0.1|1天|3.78157|3.70971|3.68627|
|Correlation|checkin|chicago|dc|0.1|3天|3.69740|3.14678|3.16276|
|Correlation|checkin|chicago|dc|0.1|5天|3.62814|3.16973|3.18279|
|Correlation|checkin|chicago|dc|0.1|7天|3.58433|3.10981|3.14764|
|Correlation|checkin|chicago|nyc|0.1|1天|7.33929|6.93074|6.98819|
|Correlation|checkin|chicago|nyc|0.1|3天|7.22262|6.81888|6.78142|
|Correlation|checkin|chicago|nyc|0.1|5天|7.13852|6.68927|7.10522|
|Correlation|checkin|chicago|nyc|0.1|7天|7.12167|6.68495|7.10102|
|Correlation|checkin|dc|nyc|0.1|1天|9.42764|7.58686|7.48810|
|Correlation|checkin|dc|nyc|0.1|3天|7.92233|6.87626|7.59136|
|Correlation|checkin|dc|nyc|0.1|5天|7.87232|7.68152|7.87473|
|Correlation|checkin|dc|nyc|0.1|7天|7.81039|7.81719|7.80975|
|Correlation|checkin|dc|chicago|0.1|1天|10.58203|4.79336|4.42591|
|Correlation|checkin|dc|chicago|0.1|3天|11.23320|3.63033|3.59785|
|Correlation|checkin|dc|chicago|0.1|5天|9.93056|4.27806|3.38808|
|Correlation|checkin|dc|chicago|0.1|7天|8.54751|4.07764|3.38041|

#### Dynamic training mode, other parameters are the same

|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Correlation|checkin|nyc|chicago|0.1|1天|4.67712|4.36085|3.98304|
|Correlation|checkin|nyc|chicago|0.1|3天|4.22121|3.71737|3.71944|
|Correlation|checkin|nyc|chicago|0.1|5天|4.32650|4.04136|4.12380|
|Correlation|checkin|nyc|chicago|0.1|7天|3.77726|3.60159|3.61911|
|Correlation|checkin|nyc|dc|0.1|1天|3.77775|3.26845|3.24397|
|Correlation|checkin|nyc|dc|0.1|3天|3.53591|3.26197|3.23575|
|Correlation|checkin|nyc|dc|0.1|5天|3.39463|3.40511|3.40644|
|Correlation|checkin|nyc|dc|0.1|7天|3.38310|3.32217|3.33183|
|Correlation|checkin|chicago|dc|0.1|1天|3.78157|3.70971|3.68629|
|Correlation|checkin|chicago|dc|0.1|3天|3.69740|3.12705|3.15397|
|Correlation|checkin|chicago|dc|0.1|5天|3.62814|3.16530|3.18082|
|Correlation|checkin|chicago|dc|0.1|7天|3.58433|3.11014|3.15014|
|Correlation|checkin|chicago|nyc|0.1|1天|7.33929|6.93074|6.98320|
|Correlation|checkin|chicago|nyc|0.1|3天|7.22262|6.81979|6.79395|
|Correlation|checkin|chicago|nyc|0.1|5天|7.13852|6.67125|7.15681|
|Correlation|checkin|chicago|nyc|0.1|7天|7.12167|6.65707|7.09085|
|Correlation|checkin|dc|nyc|0.1|1天|9.42764|7.58686|7.67655|
|Correlation|checkin|dc|nyc|0.1|3天|7.92233|6.69994|9.09571|
|Correlation|checkin|dc|nyc|0.1|5天|7.87232|7.65244|7.86764|
|Correlation|checkin|dc|nyc|0.1|7天|7.81039|7.41190|7.80580|
|Correlation|checkin|dc|chicago|0.1|1天|10.58203|4.79336|4.65393|
|Correlation|checkin|dc|chicago|0.1|3天|11.23320|3.68010|4.08812|
|Correlation|checkin|dc|chicago|0.1|5天|9.93056|3.95342|3.64199|
|Correlation|checkin|dc|chicago|0.1|7天|8.54751|4.06561|3.93779|

#### The same parameter, except a smaller lr

|    Graph    | Match        |   SD    |   TD    | transfer-ratio | TD-训练样本数量 | TD-Direct |  TD-FT  | TD-Transfer |
| :---------: | ------------ | :-----: | :-----: | :------------: | :-------------: | :-------: | :-----: | :---------: |
|Correlation|checkin|nyc|chicago|0.1|1天|4.67712|3.92269|3.99061|
|Correlation|checkin|nyc|chicago|0.1|3天|4.22121|3.71275|3.74433|
|Correlation|checkin|nyc|chicago|0.1|5天|4.32650|3.87218|4.06787|
|Correlation|checkin|nyc|chicago|0.1|7天|3.77726|3.62829|3.64411|
|Correlation|checkin|nyc|dc|0.1|1天|3.77775|3.41202|3.24368|
|Correlation|checkin|nyc|dc|0.1|3天|3.53591|3.23791|3.22884|
|Correlation|checkin|nyc|dc|0.1|5天|3.39463|3.39410|3.39393|
|Correlation|checkin|nyc|dc|0.1|7天|3.38310|3.23871|3.23573|
|Correlation|checkin|chicago|dc|0.1|1天|3.78157|3.71789|3.69317|
|Correlation|checkin|chicago|dc|0.1|3天|3.69740|3.13670|3.16728|
|Correlation|checkin|chicago|dc|0.1|5天|3.62814|3.16764|3.18577|
|Correlation|checkin|chicago|dc|0.1|7天|3.58433|3.11301|3.15291|
|Correlation|checkin|chicago|nyc|0.1|1天|7.33929|6.90001|6.96449|
|Correlation|checkin|chicago|nyc|0.1|3天|7.22262|6.79849|6.76211|
|Correlation|checkin|chicago|nyc|0.1|5天|7.13852|6.69655|7.10062|
|Correlation|checkin|chicago|nyc|0.1|7天|7.12167|6.69928|7.10788|
|Correlation|checkin|dc|nyc|0.1|1天|9.42764|7.78120|7.75579|
|Correlation|checkin|dc|nyc|0.1|3天|7.92233|6.91918|8.86747|
|Correlation|checkin|dc|nyc|0.1|5天|7.87232|7.59995|7.87427|
|Correlation|checkin|dc|nyc|0.1|7天|7.81039|7.43801|7.81107|
|Correlation|checkin|dc|chicago|0.1|1天|10.58203|4.67753|4.75832|
|Correlation|checkin|dc|chicago|0.1|3天|11.23320|4.37880|4.00140|
|Correlation|checkin|dc|chicago|0.1|5天|9.93056|3.78431|3.73602|
|Correlation|checkin|dc|chicago|0.1|7天|8.54751|4.07984|3.84819|

## Experiment 1

Parameters