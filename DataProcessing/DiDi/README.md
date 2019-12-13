#### DiDi Data

**Step 1**, download data from DiDi, and put the data into the DiDiData dir.

```python
xian: (download both)
    https://outreach.didichuxing.com/app-vue/XiAnOct2016?id=6
    https://outreach.didichuxing.com/app-vue/XiAnNov2016?id=5
chengdu: (download both)
    https://outreach.didichuxing.com/app-vue/ChengDuOct2016?id=4
    https://outreach.didichuxing.com/app-vue/personal?id=1
```

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

