# _*_ coding:utf-8 _*_
import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
                 '--Dataset DiDi '
                 '--GLL 1 '
                 '--LSTMUnits 64 '
                 '--GALUnits 64 '
                 '--GALHeads 2 '
                 '--DenseUnits 32 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0.65 '
                 '--TD 7500 '
                 '--TI 30 '
                 '--Epoch 10000 '
                 '--Train False '
                 '--lr 5e-5 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 500 '
                 '--BatchSize 128 '
                 '--Device 1 ')

if __name__ == "__main__":

    # 可以先选择在 DiDi-Xian, DiDi-Chengdu, Metro-Shanghai, ChargeStation-Beijing 这几个数据集上进行测试，因为耗时比较短

    # stability test
    test_times = 10
    for i in range(test_times):
        os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chengdu --Group Chengdu'
                                  ' --K 1 --L 1 --Graph Distance-Interaction-Correlation --CodeVersion STA%s' % i)