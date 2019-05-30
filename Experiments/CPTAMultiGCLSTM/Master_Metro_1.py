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
                 '--TC 0 '
                 '--TD 1000 '
                 '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 5e-4 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 512 ')

if __name__ == "__main__":
    
    # os.system(shared_params + '--CT 6 --PT 7 --TT 4 --K 0 --L 1 --Device 0 '
    #                           '--City Chongqing --Group ChongqingTest --Graph Distance --CodeVersion VN4')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --K 1 --L 1 --Device 0 '
                              '--City Chongqing --Group ChongqingTest --Graph line --CodeVersion VN4')

    os.system(shared_params + '--CT 6 --PT 7 --TT 4 --K 1 --L 1 --Device 0 '
                              '--City Chongqing --Group ChongqingTest --Graph transfer --CodeVersion VN4')