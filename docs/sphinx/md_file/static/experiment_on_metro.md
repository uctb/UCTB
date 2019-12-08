## Experiments on subway traffic-flow prediction

- Experiment Setting

  - Dataset

    |        Attributes        |  **Chongqing**  |  **Shanghai**   |
    | :----------------------: | :-------------: | :-------------: |
    |        Time span         | 2016.08-2017.07 | 2017.07-2017.09 |
    | Number of riding records |    409277117    |    403071370    |
    |    Number of stations    |       113       |       288       |
    |     Number of lines      |        5        |       14        |

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
    |     lr     | 3e-4  |               learning rate                |

- Experiment Results

  - STMeta uses correlation graph and neighbor graph.

|               | Chongqing |  Shanghai  |
| :-----------: | :-------: | :--------: |
|      HM       | 786.01197 | 1247.56662 |
|     ARIMA     | 660.28378 | 967.16123  |
|      HMM      | 660.28378 | 614.19177  |
|    XGBoost    | 289.70050 | 416.58629  |
|     LSTM      | 239.97653 | 408.09871  |
| STMeta | 138.81463 | 251.38817  |

#### Adding period and closeness feature into mode

|               | Chongqing | Shanghai |
| :-----------: | :-------: | :------: |
|      HM       | 227.0985  |          |
|     ARIMA     |           |          |
|    XGBoost    | 134.9760  |          |
|     GBRT      | 120.3337  |          |
|     LSTM      | 124.6012  |          |
|   DG-GCLSTM   |           |          |
|   IG-GCLSTM   |           |          |
|   CG-GCLSTM   |           |          |
| STMeta |           |          |

<u>[Back To HomePage](../index.html)</u>

