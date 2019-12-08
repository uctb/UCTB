## Experiments on DiDi traffic-flow prediction

- Experiment Setting

  - Dataset

    |        Attributes        |    **Xi'an**    |   **Chengdu**   |
    | :----------------------: | :-------------: | :-------------: |
    |        Time span         | 2016.10-2016.11 | 2016.10-2016.11 |
    | Number of riding records |     5922891     |     8439537     |
    |    Number of stations    |       253       |       256       |

  In the data preprocessing stage, we removed the stations with average daily traffic flow smaller than 1, since predictions for these stations are not significant in real life.

  Network parameter setting (STMeta model)

  - Following shows the parameters we used and a short explanation of the parameter meaning.  To know more about the parameter, please refer to the API reference.

    | Parameter  | Value |                  Meaning                   |
    | :--------: | :---: | :----------------------------------------: |
    |    GLL     |   2   |          number of GCLSTM layers           |
    | LSTMUnits  |  256  |       number of hidden units in LSTM       |
    |  GALUnits  |  256  |       number of units used in GAtteL       |
    |  GALHeads  |   2   |       number of multi-head in GAtteL       |
    | DenseUnits |  32   | number of units in penultimate dense layer |
    |     TC     |   0   |           correlation threshold            |
    |     TD     | 1000  |             distance threshold             |
    |     TI     |  500  |           interaction threshold            |
    |     lr     | 5e-4  |               learning rate                |

- Experiment Results

  - STMeta uses correlation graph and interaction graph on Xi'an dataset, and uses correlation graph and interaction graph on Chengdu dataset.

|               |  Xi'an   | Chengdu  |
| :-----------: | :------: | :------: |
|      HM       | 10.13594 | 14.14511 |
|     ARIMA     | 10.11464 | 14.53216 |
|      HMM      | 10.3239  | 15.24771 |
|    XGBoost    | 8.72033  | 10.73894 |
|     LSTM      | 9.31375  | 12.05271 |
| STMeta | 7.20868  | 8.88920  |

#### Adding period and closeness feature into mode

|               |  Xi'an  | Chengdu |
| :-----------: | :-----: | :-----: |
|      HM       |         |         |
|     ARIMA     |         |         |
|    XGBoost    |         |         |
|     GBRT      |         |         |
|     LSTM      | 6.33135 | 6.97385 |
|   DG-GCLSTM   | 6.50630 | 7.16565 |
|   IG-GCLSTM   | 6.02866 | 7.20899 |
|   CG-GCLSTM   | 5.90418 | 7.09994 |
| STMeta | 5.94591 | 7.26551 |

<u>[Back To HomePage](../index.html)</u>