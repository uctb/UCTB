import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python DCRNN.py '
                         '--Dataset Metro '
                         '--CT 6 '
                         '--PT 0 '
                         '--TT 0 '
                         '--K 1 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 1 '
                         '--DataRange All '
                         '--TrainDays 365 '
                         '--TC 0.7 '
                         '--TD 5000 '
                         '--TI 30 '
                         '--Epoch 20000 '
                         '--Train True '
                         '--lr 1e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--Device 1 '
                         '--CodeVersion V0')

if __name__ == "__main__":

    os.system(shared_params_st_mgcn + ' --City Shanghai --Graph Distance --MergeIndex 3')

    os.system(shared_params_st_mgcn + ' --City Chongqing --Graph Distance --MergeIndex 3')

    os.system(shared_params_st_mgcn + ' --City Shanghai --Graph Distance --MergeIndex 6')

    os.system(shared_params_st_mgcn + ' --City Chongqing --Graph Distance --MergeIndex 6')

    os.system(shared_params_st_mgcn + ' --City Shanghai --Graph Distance --MergeIndex 12')

    os.system(shared_params_st_mgcn + ' --City Chongqing --Graph Distance --MergeIndex 12')