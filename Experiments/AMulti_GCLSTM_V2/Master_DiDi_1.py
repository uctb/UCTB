import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python AMulti_GCLSTM_V2_Obj.py '
                 '--Dataset DiDi '
                 '--CT 6 '
                 '--PT 7 '
                 '--TT 4 '
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
                 '--Train True '
                 '--lr 5e-5 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 500 '
                 '--BatchSize 128 '
                 '--Device 1 '
                 '--CodeVersion V2 ')

if __name__ == "__main__":

    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 0 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Correlation')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Distance-Correlation')