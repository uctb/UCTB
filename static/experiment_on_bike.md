## Experiments on bike traffic-flow prediction

- Experiment Setting

  - Dataset

    |        Attributes        | **New York City** |   **Chicago**   |     **DC**      |
    | :----------------------: | :---------------: | :-------------: | :-------------: |
    |        Time span         |  2013.03-2017.09  | 2013.06-2017.12 | 2013.07-2017.09 |
    | Number of riding records |    49,669,470     |   13,826,196    |   13,763,675    |
    |    Number of stations    |        827        |       586       |       531       |

  Following shows a map-visualization of bike stations in NYC, the latest built stations have deeper color.

  <img src="../src/image/Bike_NYC_STMAP.PNG">

  In the data preprocessing stage, we removed the stations with average daily traffic flow smaller than 1, since predictions for these stations are not significant in real life. The remaining station number are 717, 444 and 378 for NYC, Chicago and DC, respectively.

  - Network parameter setting (AMulti-GCLSTM model)

    Following shows the parameters we used and a short explanation of the parameter meaning.  To know more about the parameter, please refer to the API reference.

- Experiment Results

  - CG-GCLSTM Only use correlation graph in the model
  - DG-GCLSTM Only use distance graph in the model
  - IG-GCLSTM Only use interaction graph in the model

#### Only closeness feature (will delete in future version)

|                       |   NYC   | Chicago |   DC    |
| :-------------------: | :-----: | :-----: | :-----: |
|          HM           | 6.79734 | 4.68078 | 3.66747 |
|         ARIMA         | 5.60477 | 3.79739 | 3.31826 |
|          HMM          | 5.42030 | 3.79743 | 3.20889 |
|        XGBoost        | 5.32069 | 3.75124 | 3.14101 |
|         LSTM          | 5.13307 | 3.69806 | 3.14331 |
|       CG-GCLSTM       | 4.64375 | 3.38255 | 2.87655 |
|       DG-GCLSTM       | 4.67169 | 3.51243 | 2.98404 |
|       IG-GCLSTM       | 4.77809 | 3.45625 | 2.68370 |
| ST_MGCN (Multi-Graph) | 4.41732 |         |         |
|     AMulti-GCLSTM     | 4.22640 | 3.02301 | 2.58584 |

#### Latest result

|                       |        NYC         |      Chicago       |         DC         |
| :-------------------: | :----------------: | :----------------: | :----------------: |
| HM（params searched） | 3.99224 (C1-P1-T3) | 2.97693 (C1-P1-T2) | 2.63165 (C2-P1-T3) |
|        XGBoost        |      4.14909       |      3.02530       |      2.73286       |
|         GBRT          |      3.94348       |      2.85847       |      2.63935       |
|         LSTM          |      3.78497       |      2.79078       |      2.54752       |
|       DG-GCLSTM       |      3.63207       |      2.71876       |      2.53863       |
|       IG-GCLSTM       |      3.78816       |      2.70131       |      2.46214       |
|       CG-GCLSTM       |      3.69656       |      2.79812       |      2.45815       |
|   AMulti-GCLSTM-V1    |      3.50475       |    **2.65511**     |      2.42582       |
|   AMulti-GCLSTM-V2    |    **3.43870**     |      2.66379       |    **2.41111**     |

- Model training records

  Following data was collected on a Windows PC with *CPU : Interl i7 8700K, Memory: 32 GB, GPU: Nvidia GTX 1080Ti*. 

|        ```NYC City```         | SingleGraph-GCLSTM(Average) | AMulti-GCLSTM | ST_MGCN  |
| :---------------------------: | :-------------------------: | :-----------: | :------: |
| Number of trainable variables |                             |               |  249245  |
|   Converaged training time    |                             |               | 51 hours |

- Source Code

Use the ```./Experiment/AMultiGCLSTM_Master_Bike.py``` to train the model or view evaluation results. 



<u>[Back To HomePage](../index.html)</u>