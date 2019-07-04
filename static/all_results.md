# Results on different datasets

## AMultiGCLSTM Version

|   Version Name   | Temporal Feature Process | Temporal Merge Method | Multi-Graph Merge Method | Parameter Complexity |
| :--------------: | :----------------------: | :-------------------: | :----------------------: | :------------------: |
| AMulti-GCLSTM-V1 |          GCLSTM          |          GAL          |           GAL            |                      |
| AMulti-GCLSTM-V2 |          GCLSTM          |     Concat+Dense      |           GAL            |                      |
| AMulti-GCLSTM-V3 |           RGAL           |          GAL          |           GAL            |                      |

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
|  AMulti-GCLSTM-V2  |      3.43870       |      2.66379       |      2.41111       |

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
|    AMulti-GCLSTM-V3     |                             |             |

## Results on Metro

|                         |  Chongqing   |   Shanghai    |
| :---------------------: | :----------: | :-----------: |
|   HM（已做参数搜索）    |   120.3116   |   197.9747    |
| XGBoost（已做参数搜索） |   118.0798   |   190.2446    |
|  GBRT（已做参数搜索）   |   122.8662   |   185.9844    |
|        CPT-LSTM         |   99.5716    |   194.5480    |
|        DG-GCLSTM        |   99.5361    |   199.1488    |
|   IG-GCLSTM （line）    |   101.1115   |   159.0944    |
|        CG-GCLSTM        |   98.5321    |   185.0774    |
|    AMulti-GCLSTM-V1     |   98.3840    |   170.7830    |
|    AMulti-GCLSTM-V2     | **96.31608** | **154.59911** |
|    AMulti-GCLSTM-V3     |              |               |

#### 以下结果待整理

```python
Metro Chongqing AMultiGCLSTM_V2_D_K0L1_V2
Number of trainable variables 58690
Test result [105.65234712535701, 0.2545523403057227]

Metro Chongqing AMultiGCLSTM_V2_D_K1L1_V2
Number of trainable variables 58693
Test result [112.33960270461257, 0.40374120883167647]

Metro Chongqing AMultiGCLSTM_V2_l_K1L1_V2
Number of trainable variables 58693
Test result [105.91747791948298, 0.25791858811799]

Metro Chongqing AMultiGCLSTM_V2_C_K1L1_V2
Number of trainable variables 58693
Test result [112.97357984219778, 0.3869267307557338]

Metro Chongqing AMultiGCLSTM_V2_DlC_K1L1_V2
Number of trainable variables 180561
Test result [96.31608920493976, 0.2666995133247486]
```

```
Metro ShanghaiV1 AMultiGCLSTM_V2_D_K0L1_V2
Number of trainable variables 58690
Test result [189.92262158079583, 0.12731130479655822]

Metro ShanghaiV1 AMultiGCLSTM_V2_D_K1L1_V2
Number of trainable variables 58693
Test result [186.0371212508715, 0.15392561472750804]

Metro ShanghaiV1 AMultiGCLSTM_V2_l_K1L1_V2
Number of trainable variables 58693
Test result [185.05749033560113, 0.18560633686281308]

Metro ShanghaiV1 AMultiGCLSTM_V2_C_K1L1_V2
Number of trainable variables 58693
Test result [184.81174520758032, 0.16507430693288]

Metro ShanghaiV1 AMultiGCLSTM_V2_DlC_K1L1_V2
Number of trainable variables 180561
Test result [154.59911616499195, 0.13682135383499028]
```

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

