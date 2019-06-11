import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
                 '--Dataset Bike '
                 '--CT 6 '
                 '--PT 7 '
                 '--TT 0 '
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
                 '--BatchSize 16 '
                 '--Device 0 '
                 '--CodeVersion V0T0 ')

if __name__ == "__main__":

    """
    Single Graph
    """
    os.system(shared_params + ' --City NYC --Group NYC --Graph Distance --K 0 --L 1')
    os.system(shared_params + ' --City NYC --Group NYC --Graph Distance --K 1 --L 1')
    os.system(shared_params + ' --City NYC --Group NYC --Graph Interaction --K 1 --L 1')
    os.system(shared_params + ' --City NYC --Group NYC --Graph Correlation --K 1 --L 1')

    os.system(shared_params + ' --City Chicago --Group Chicago --Graph Distance --K 0 --L 1')
    os.system(shared_params + ' --City Chicago --Group Chicago --Graph Distance --K 1 --L 1')
    os.system(shared_params + ' --City Chicago --Group Chicago --Graph Interaction --K 1 --L 1')
    os.system(shared_params + ' --City Chicago --Group Chicago --Graph Correlation --K 1 --L 1')

    os.system(shared_params + ' --City DC --Group DC --Graph Distance --K 0 --L 1')
    os.system(shared_params + ' --City DC --Group DC --Graph Distance --K 1 --L 1')
    os.system(shared_params + ' --City DC --Group DC --Graph Interaction --K 1 --L 1')
    os.system(shared_params + ' --City DC --Group DC --Graph Correlation --K 1 --L 1')