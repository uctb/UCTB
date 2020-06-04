import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python DCRNN.py '
                         '--Dataset Bike '
                         '--CT 6 '
                         '--PT 0 '
                         '--TT 0 '
                         '--K 1 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 1 '
                         '--DataRange All '
                         '--TrainDays 365 '
                         '--TC 0 '
                         '--TD 1000 '
                         '--TI 500 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 5e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 32 '
                         '--Device 0 '
                         '--CodeVersion V0')

if __name__ == "__main__":
    os.system(shared_params_st_mgcn + ' --City NYC --Graph Distance --MergeIndex 6')

    # os.system(shared_params_st_mgcn + ' --City DC --Graph Distance --MergeIndex 6')

    # os.system(shared_params_st_mgcn + ' --City Chicago --Graph Distance --MergeIndex 6')