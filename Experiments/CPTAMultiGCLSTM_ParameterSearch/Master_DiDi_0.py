import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
                 '--Dataset DiDi '
                 '--CT 6 '
                 '--PT 3 '
                 '--TT 0 '
                 '--GLL 1 '
                 '--LSTMUnits 126 '
                 '--GALUnits 64 '
                 '--GALHeads 2 '
                 '--DenseUnits 1024 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0 '
                 '--TD 2000 '
                 '--TI 200 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 5e-5 '
                 '--Normalize False '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 128 '
                 '--Device 2 '
                 '--CodeVersion V0 ')

if __name__ == "__main__":

    os.system(shared_params + ' --City Xian --Group Xian --K 0 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Correlation')
    os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Interaction')
    os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Distance-Interaction-Correlation')

    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 0 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Correlation')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Interaction')
    os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Distance-Interaction-Correlation')
