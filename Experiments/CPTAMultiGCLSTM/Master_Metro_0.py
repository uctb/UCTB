import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj_Metro.py '
                 '--Dataset Metro '
                 '--GLL 1 '
                 '--LSTMUnits 126 '
                 '--GALUnits 64 '
                 '--GALHeads 2 '
                 '--DenseUnits 1024 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0 '
                 '--TD 1000 '
                 '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 5e-5 '
                 '--Normalize False '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 64 '
                 '--Device 2 ')

if __name__ == "__main__":

    # Chongqing

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 0 --L 1 --Graph Distance --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Distance --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Correlation --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph line --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Distance-Correlation-line --CodeVersion V0')

    # Shanghai

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
                              ' --K 0 --L 1 --Graph Distance --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
                              ' --K 1 --L 1 --Graph Distance --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
                              ' --K 1 --L 1 --Graph Correlation --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
                              ' --K 1 --L 1 --Graph line --CodeVersion V0')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
                              ' --K 1 --L 1 --Graph Distance-Correlation-line --CodeVersion V0')
