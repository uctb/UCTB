import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
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
                 '--BatchSize 128 '
                 '--Device 0 ')

if __name__ == "__main__":
    
    os.system(shared_params + '--CT 0 --PT 7 --TT 0 --City Chongqing --Group ChongqingTest --K 0 --L 1 --Graph Distance --CodeVersion V1OP')

