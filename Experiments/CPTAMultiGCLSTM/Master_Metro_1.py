import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj_Metro.py '
                 '--Dataset Metro '
                 '--GLL 1 '
                 '--LSTMUnits 128 '
                 '--GALUnits 128 '
                 '--GALHeads 2 '
                 '--DenseUnits 64 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0 '
                 '--TD 2000 '
                 '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 1e-4 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 64 '
                 '--Device 1 '
                 '--CodeVersion VN7 ')

if __name__ == "__main__":

    # Chongqing

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 0 --L 1 --Graph Distance')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Distance')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Correlation')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph line')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph neighbor')

    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
    #                           ' --K 1 --L 1 --Graph Distance-Correlation-line')

    # Shanghai

    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
    #                           ' --K 0 --L 1 --Graph Distance')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
    #                           ' --K 1 --L 1 --Graph Distance')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
    #                           ' --K 1 --L 1 --Graph Correlation')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
    #                           ' --K 1 --L 1 --Graph line')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Shanghai --Group Shanghai'
    #                           ' --K 1 --L 1 --Graph Distance-Correlation-line')