import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
                 '--Dataset Metro '
                 '--CT 0 '
                 '--PT 7 '
                 '--TT 4 '
                 '--GLL 1 '
                 '--LSTMUnits 64 '
                 '--GALUnits 64 '
                 '--GALHeads 2 '
                 '--PTALUnits 128 '
                 '--PTALHeads 2 '
                 '--DenseUnits 32 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0 '
                 '--TD 1000 '
                 '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 1e-4 '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 128 '
                 '--Device 1 '
                 '--CodeVersion V1 ')

if __name__ == "__main__":

    """
    Multiple Graphes
    """
    os.system(shared_params + ' --City Chongqing --Group Chongqing --K 1 --L 1'
                              ' --Graph Distance-Correlation')