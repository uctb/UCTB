import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python DCRNN.py '
                         '--Dataset METR '
                         '--CT 6 '
                         '--PT 0 '
                         '--TT 0 '
                         '--K 1 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 1 '
                         '--DataRange All '
                         '--TrainDays All '
                         '--TC 0.7 '
                         '--TD 5500 '
                         '--TI 30 '
                         '--Epoch 20000 '
                         '--test_ratio 0.2 '
                         '--Train True '
                         '--lr 1e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--MergeWay average '
                         '--Device 1 '
                         '--CodeVersion V0')

if __name__ == "__main__":

    os.system(shared_params_st_mgcn + ' --City LA --Graph Distance --MergeIndex 3')

    os.system(shared_params_st_mgcn + ' --City LA --Graph Distance --MergeIndex 6')

    os.system(shared_params_st_mgcn + ' --City LA --Graph Distance --MergeIndex 12')

  