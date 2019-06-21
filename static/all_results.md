# Results on different datasets

## AMultiGCLSTM Version

|   Version Name   |            Temporal Feature Process             | Temporal Merge Method | Multi-Graph Merge Method | Parameter Complexity |
| :--------------: | :---------------------------------------------: | :-------------------: | :----------------------: | -------------------- |
| AMulti-GCLSTM-V0 | Closeness-GCLSTM, Period/Trend - AttentionMerge |     Concat+Dense      |           GAL            |                      |
| AMulti-GCLSTM-V1 |                     GCLSTM                      |       GAL+Dense       |           GAL            |                      |
| AMulti-GCLSTM-V2 |                     GCLSTM                      |     Concat+Dense      |           GAL            |                      |

##### 以下实验中，除了标记了 （已做参数搜索），其他均未做参数搜索

## Results on Bike

|                    |        NYC         |      Chicago       |         DC         |
| :----------------: | :----------------: | :----------------: | :----------------: |
| HM（已做参数搜索） | 3.99224 (C1-P1-T3) | 2.97693 (C1-P1-T2) | 2.63165 (C2-P1-T3) |
|      XGBoost       |      4.14909       |      3.02530       |      2.73286       |
|        GBRT        |      3.94348       |      2.85847       |      2.63935       |
|        LSTM        |      3.78497       |      2.79078       |      2.54752       |
|     DG-GCLSTM      |      3.63207       |      2.71876       |      2.53863       |
|     IG-GCLSTM      |      3.78816       |      2.70131       |      2.46214       |
|     CG-GCLSTM      |      3.69656       |      2.79812       |      2.45815       |
|  AMulti-GCLSTM-V1  |    **3.50475**     |    **2.65511**     |    **2.42582**     |

## Results on DiDi

|                         |            Xi'an            |   Chengdu   |
| :---------------------: | :-------------------------: | :---------: |
|   HM（已做参数搜索）    |           6.19809           |   7.35477   |
| XGBoost（已做参数搜索） |           6.58816           |   7.81596   |
|  GBRT（已做参数搜索）   |           8.15679           |   7.58014   |
|          LSTM           |                             |             |
|        DG-GCLSTM        |                             |             |
|        IG-GCLSTM        |                             |             |
|        CG-GCLSTM        |                             |             |
|    AMulti-GCLSTM-V1     | **5.88538**（已做参数搜索） |   7.13728   |
|    AMulti-GCLSTM-V2     |           5.90493           | **7.03252** |

## Results on Metro

|                         |  Chongqing  |   Shanghai    |
| :---------------------: | :---------: | :-----------: |
|   HM（已做参数搜索）    |  120.3116   |   197.9747    |
| XGBoost（已做参数搜索） |  118.0798   |   190.2446    |
|  GBRT（已做参数搜索）   |  122.8662   |   185.9844    |
|        CPT-LSTM         |   99.5716   |   194.5480    |
|        DG-GCLSTM        |   99.5361   |   199.1488    |
|   IG-GCLSTM （line）    |  101.1115   |   159.0944    |
|        CG-GCLSTM        |   98.5321   |   185.0774    |
|    AMulti-GCLSTM-V1     | **98.3840** |   170.7830    |
|    AMulti-GCLSTM-V2     |             | **157.26091** |

## Results on Charge-Station

|                  |   Beijing   |
| :--------------: | :---------: |
|        HM        |   1.13594   |
|     XGBoost      |   0.98561   |
|       GBRT       |   0.98317   |
|     CPT-LSTM     |   0.81497   |
|    DG-GCLSTM     |   0.83583   |
|    CG-GCLSTM     | **0.80491** |
| AMulti-GCLSTM-V1 |   0.81599   |
| AMulti-GCLSTM-V2 |   0.81915   |
