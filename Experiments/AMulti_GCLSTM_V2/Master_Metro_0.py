import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python AMulti_GCLSTM_V2_Obj.py '
                 '--Dataset Metro '
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
                 '--TC 0.7 '
                 '--TD 5000 '
                 # '--TI 30 '
                 '--Epoch 20000 '
                 '--Train True '
                 '--lr 2e-5 '
                 '--Normalize True '
                 '--patience 0.1 '
                 '--ESlength 500 '
                 '--BatchSize 128 '
                 '--Device 0 '
                 '--CodeVersion V2_1 ')

if __name__ == "__main__":

    # Chongqing

    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 0 --L 1 --Graph Distance')
    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph Distance')
    os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph line')
    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph Correlation')
    #
    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph Distance-line-Correlation')