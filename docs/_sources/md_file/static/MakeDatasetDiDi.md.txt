#### DiDi Data

We are especially grateful for the data provided by the DiDi Chuxing GAIA Initiative.

This dataset is from the trajectory data of DiDi Express and DiDi Premier drivers within the Second Ring Road of Xi'an and Chengdu City. The measurement interval of the track points is approximately 2-4 seconds . The track points were bound to physical roads so that the trajectory data and the actual road information are matched. The driver and trip order information were encrypted and anonymized. 

**Step 1**, download data from [https://outreach.didichuxing.com/appEn-vue/dataList](https://outreach.didichuxing.com/appEn-vue/dataList), and put the data into the DiDiData dir.

Xi'an City: (download both)<br>
    [Oct 2016, Xi’an City Second Ring Road Regional Trajectory Data Set](https://outreach.didichuxing.com/appEn-vue/XiAnOct2016?id=8) <br>    [Nov 2016, Xi’an City Second Ring Road Regional Trajectory Data Set](https://outreach.didichuxing.com/appEn-vue/XiAnNov2016?id=9) <br>Chengdu City: (download both)<br>
    [Oct 2016, Chengdu City Second Ring Road Regional Trajectory Data Set](https://outreach.didichuxing.com/appEn-vue/ChengDuOct2016?id=7) <br>
    [Nov 2016, Chengdu City Second Ring Road Regional Trajectory Data Set](https://outreach.didichuxing.com/appEn-vue/personal?id=2) 

After step1, you will have the following file-tree:

```
├── DiDiData
│   ├── chengdu
│   │   ├── chengdu_gps_20161001.json
│   │   ├── chengdu_gps_20161002.json
│   │   ├── ...
│   │   └── chengdu_gps_20161130.json
│   └── xian
│       ├── xian_gps_20161001.json
│       ├── xian_gps_20161002.json
│       ├── ...
│       └── xian_gps_20161130.json
├── release_data_dir
├── create_release_data_didi.py
├── data_config.py
├── get_grid_data_didi.py
├── get_monthly_interaction_didi.py
├── local_path.py
└── multi_threads.py
```

**Step 2**, run the following codes

```python
# build the grid data
python get_grid_data_didi.py --data DiDi --city Xian --jobs 1
python get_grid_data_didi.py --data DiDi --city Chengdu --jobs 1

# build the monthly interaction data
python get_monthly_interaction_didi.py --data DiDi --city Xian --jobs 1
python get_monthly_interaction_didi.py --data DiDi --city Chengdu --jobs 1

# Output the final file
python create_release_data_didi.py --data DiDi --city Xian
python create_release_data_didi.py --data DiDi --city Chengdu
```

`jobs` is the number of threads used by the program. A larger `jobs` will reduce the running time significantly.

You will see the `.pkl` data in `release_data_dir` after step 2.

