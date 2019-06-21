import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python CPT_AMulti_GCLSTM_Obj.py '
                 '--Dataset Bike '
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
                 '--TC 0 '
                 '--TD 1000 '
                 '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 1e-4 '
                 '--Normalize False '
                 '--patience 0.1 '
                 '--ESlength 50 '
                 '--BatchSize 64 '
                 '--Device 1 '
                 '--CodeVersion V0 ')

if __name__ == "__main__":

    os.system(shared_params + ' --City NYC --Group NYC_CPT_Test --K 0 --L 1 --Graph Distance')
    os.system(shared_params + ' --City NYC --Group NYC_CPT_Test --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City NYC --Group NYC_CPT_Test --K 1 --L 1 --Graph Correlation')
    os.system(shared_params + ' --City NYC --Group NYC_CPT_Test --K 1 --L 1 --Graph Interaction')
    
    os.system(shared_params + ' --City Chicago --Group Chicago_CPT_Test --K 0 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Chicago --Group Chicago_CPT_Test --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City Chicago --Group Chicago_CPT_Test --K 1 --L 1 --Graph Correlation')
    os.system(shared_params + ' --City Chicago --Group Chicago_CPT_Test --K 1 --L 1 --Graph Interaction')

    os.system(shared_params + ' --City DC --Group DC_CPT_Test --K 0 --L 1 --Graph Distance')
    os.system(shared_params + ' --City DC --Group DC_CPT_Test --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City DC --Group DC_CPT_Test --K 1 --L 1 --Graph Correlation')
    os.system(shared_params + ' --City DC --Group DC_CPT_Test --K 1 --L 1 --Graph Interaction')