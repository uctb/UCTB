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

    | Parameter  | Value |                  Meaning                   |
    | :--------: | :---: | :----------------------------------------: |
    |    GLL     |   1   |          number of GCLSTM layers           |
    | LSTMUnits  |  64   |       number of hidden units in LSTM       |
    |  GALUnits  |  64   |       number of units used in GAtteL       |
    |  GALHeads  |   2   |       number of multi-head in GAtteL       |
    | DenseUnits |  32   | number of units in penultimate dense layer |
    |     TC     |   0   |           correlation threshold            |
    |     TD     | 1000  |             distance threshold             |
    |     TI     |  500  |           interaction threshold            |

- Experiment Results

  - CG-GCLSTM Only use correlation graph in the model
  - DG-GCLSTM Only use distance graph in the model
  - IG-GCLSTM Only use interaction graph in the model

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

- Model training records

  Following data was collected on a Windows PC with *CPU : Interl i7 8700K, Memory: 32 GB, GPU: Nvidia GTX 1080Ti*. 

|        ```NYC City```         | SingleGraph-GCLSTM(Average) | AMulti-GCLSTM | ST_MGCN  |
| :---------------------------: | :-------------------------: | :-----------: | :------: |
| Number of trainable variables |            19749            |     61993     |  249245  |
|   Converaged training time    |           2 hours           |    6 hours    | 51 hours |

- Source Code

Use the ```./Experiment/AMultiGCLSTM_Master_Bike.py``` to train the model or view evaluation results. 

