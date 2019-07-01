import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python AMulti_GCLSTM_V0_Obj.py '
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
                 '--TC 0.6 '
                 '--TD 3000 '
                 '--TI 50 '
                 '--Epoch 5000 '
                 '--Train False '
                 '--lr 1e-4 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 200 '
                 '--BatchSize 128 '
                 '--Device 1 '
                 '--CodeVersion TNT0 ')

if __name__ == "__main__":
<<<<<<< HEAD
    # os.system(shared_params + ' --City Xian --Group Xian --K 0 --L 1 --Graph Distance')
    # os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Correlation')
    # os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Interaction')
    # os.system(shared_params + ' --City Xian --Group Xian --K 1 --L 1 --Graph Distance-Interaction-Correlation')
=======
>>>>>>> f1cd6151a8ed652dd84b96bbee0e90de857dc4c7

    # os.system(shared_params + ' --City Chengdu --Group Chengdu --K 0 --L 1 --Graph Distance')
    # os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Distance')
    # os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Correlation')
    # os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Interaction')
    # os.system(shared_params + ' --City Chengdu --Group Chengdu --K 1 --L 1 --Graph Distance-Interaction-Correlation')