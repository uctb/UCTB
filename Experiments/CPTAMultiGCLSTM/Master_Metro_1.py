import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj_Metro.py '
                 '--Dataset Metro '
                 '--GLL 1 '
                 '--LSTMUnits 64 '
                 '--GALUnits 64 '
                 '--GALHeads 2 '
                 '--DenseUnits 32 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0.7 '
                 '--TD 5000 '
                 # '--TI 500 '
                 '--Epoch 10000 '
<<<<<<< HEAD
                 '--Train True '
=======
                 '--Train False '
>>>>>>> f1cd6151a8ed652dd84b96bbee0e90de857dc4c7
                 '--lr 1e-4 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 200 '
                 '--BatchSize 256 '
<<<<<<< HEAD
                 '--Device 1 '
                 '--CodeVersion TNT0 ')
=======
                 '--Device 0 '
                 '--CodeVersion TNT1 ')
>>>>>>> f1cd6151a8ed652dd84b96bbee0e90de857dc4c7

if __name__ == "__main__":

    # Chongqing

<<<<<<< HEAD
    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 0 --L 1 --Graph Distance')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Distance')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph Correlation')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
                              ' --K 1 --L 1 --Graph line')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
=======
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
    #                           ' --K 0 --L 1 --Graph Distance')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
    #                           ' --K 1 --L 1 --Graph Distance')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
    #                           ' --K 1 --L 1 --Graph Correlation')
    #
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
    #                           ' --K 1 --L 1 --Graph line')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --City Chongqing --Group Chongqing'
>>>>>>> f1cd6151a8ed652dd84b96bbee0e90de857dc4c7
                              ' --K 1 --L 1 --Graph Distance-Correlation-line')