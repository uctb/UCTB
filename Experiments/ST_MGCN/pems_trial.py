import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN_Obj.py '
                         '--Dataset PEMS '
                         '--CT 6 '
                         '--PT 7 '
                         '--TT 4 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
                         '--DataRange All '
                         '--TrainDays All '
                         '--TC 0.73 '
                         '--TD 5500 '
                         '--TI 30 '
                         '--Epoch 10000 '
                         '--test_ratio 0.2 '
                         '--Train True '
                         '--lr 1e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--MergeWay average '
                         '--Device 1 ')

if __name__ == "__main__":

    """
    Multiple Graphs
    """
    os.system(shared_params_st_mgcn + ' --City BAY --K 1 --L 1  '
                                      ' --Graph Distance-Correlation --MergeIndex 1')

    os.system(shared_params_st_mgcn + ' --City BAY --K 1 --L 1  '
                                      ' --Graph Distance-Correlation --MergeIndex 3')

    os.system(shared_params_st_mgcn + ' --City BAY --K 1 --L 1  '
                                      ' --Graph Distance-Correlation --MergeIndex 6')

    os.system(shared_params_st_mgcn + ' --City BAY --K 1 --L 1  '
                                      ' --Graph Distance-Correlation --MergeIndex 12')
