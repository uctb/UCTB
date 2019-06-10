import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
                 '--Dataset DiDi '
                 '--CT 6 '
                 '--PT 7 '
                 '--TT 4 '
                 '--GLL 1 '
                 '--LSTMUnits 128 '
                 '--GALUnits 128 '
                 '--GALHeads 2 '
                 '--DenseUnits 64 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0 '
                 '--TD 1000 '
                 '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 1e-4 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 128 '
                 '--Device 0 '
                 '--CodeVersion VN1 ')

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
