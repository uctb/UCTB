import os

import warnings
warnings.filterwarnings("ignore")

shared_params = ('python AMulti_GCLSTM_V0_Obj.py '
                 '--Dataset Metro '
                 '--CT 6 '
                 '--PT 7 '
                 '--TT 4 '
                 '--GLL 1 '
                 '--LSTMUnits 64 '
                 '--GALUnits 64 '
                 '--GALHeads 2 '
                 '--PTALUnits 64 '
                 '--PTALHeads 2 '
                 '--DenseUnits 32 '
                 '--DataRange All '
                 '--TrainDays All '
                 '--TC 0.7 '
                 '--TD 5000 '
                 # '--TI 500 '
                 '--Epoch 5000 '
                 '--Train True '
                 '--lr 2e-5 '
                 '--patience 0.1 '
                 '--Normalize True '
                 '--ESlength 50 '
                 '--BatchSize 128 '
                 '--Device 0 ')

if __name__ == "__main__":

    """
    Multiple Graphes
    """
    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 0 --L 1 --Graph Distance')
    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph Distance')
    # os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph Correlation')
    for i in range(10):
        os.system(shared_params + ' --City ShanghaiV1 --Group Shanghai --K 1 --L 1 --Graph Distance-Correlation'
                                  ' --CodeVersion V0_%s' % i)