#### DiDi TTI Data

We are especially grateful for the data provided by the DiDi Chuxing GAIA Initiative.

This dataset provides travel time index data for cities in Shenzhen, Chengdu, Xi'an, Suzhou, Ji'nan and Haikou in 2018, including city level, district level, road level, and average driving speed. 

**Step 1**, select the `城市交通指数(新)` dataset from [https://outreach.didichuxing.com/app-vue/DataList](https://outreach.didichuxing.com/app-vue/DataList),  download TTI data for Shenzhen, Chengdu, Xi'an, Suzhou, Ji'nan and Haikou, and unzip the data(including road.zip in every  subdirectory) into the DiDiData dir.

After step1, you will have the following file-tree:

```
├─DiDiData
│  ├─成都市
│  │      .DS_Store
│  │      boundary.txt
│  │      city_district.txt
│  │      readME.txt
│  │      成都市.txt
│  │
│  ├─济南市
│  │      .DS_Store
│  │      boundary.txt
│  │      city_district.txt
│  │      readME.txt
│  │      济南市.txt
│  │
│  ├─海口市
│  │      .DS_Store
│  │      boundary.txt
│  │      city_district.txt
│  │      readME.txt
│  │      海口市.txt
│  │
│  ├─深圳市
│  │      .DS_Store
│  │      boundary.txt
│  │      city_district.txt
│  │      readME.txt
│  │
│  ├─苏州市
│  │      .DS_Store
│  │      boundary.txt
│  │      city_district.txt
│  │      readME.txt
│  │      苏州市.txt
│  │
│  └─西安市
│          .DS_Store
│          boundary.txt
│          city_district.txt
│          readME.txt
│          西安市.txt
│
└─release_data_dir
│        README.md
│
│  DiDiTTI_utils.py
│  processingDiDiTTI.ipynb
│  README.md
│  StrictDataFormat.py
```

**Step 2**, use jupyter notebook to open [processingDiDiTTI.ipynb](../../../DataProcessing/DiDi_TTI/processingDiDiTTI.ipynb), and follow the instructions in the notebook.

Following are the details to note:

1. For missing values in the dataset, we use the data from the previous week or the next week to fill.
2.  According to [TTI Calculation](https://github.com/didi/TrafficIndex), when the road is congested, the actual speed is very slow and TTI value will become abnormally large, so we take  the reciprocal of TTI value to make the pattern more obvious.

You will see the `.pkl` data in `release_data_dir` after step 2.

