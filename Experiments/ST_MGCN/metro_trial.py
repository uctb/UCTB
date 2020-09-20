import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN_Obj.py '
                         '--Dataset Metro '
                         '--CT 6 '
                         '--PT 7 '
                         '--TT 4 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
                         '--DataRange All '
                         '--TrainDays All '
                         '--TC 0.7 '
                         '--TD 5000 '
                         '--TI 30 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 1e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--MergeWay sum '
                         '--Device 1 ')

if __name__ == "__main__":

    """
    Multiple Graphs
    """
    os.system(shared_params_st_mgcn + ' --City Shanghai --K 1 --L 1  '
                                      ' --Graph Distance-Correlation-Line --MergeIndex 3')

    os.system(shared_params_st_mgcn + ' --City Shanghai --K 1 --L 1  '
                                      ' --Graph Distance-Correlation-Line --MergeIndex 6')

    os.system(shared_params_st_mgcn + ' --City Shanghai --K 1 --L 1  '
                                      ' --Graph Distance-Correlation-Line --MergeIndex 12')

    os.system(shared_params_st_mgcn + ' --City Chongqing --K 1 --L 1  '
                                      ' --Graph Distance-Correlation-Line --MergeIndex 3')

    os.system(shared_params_st_mgcn + ' --City Chongqing --K 1 --L 1  '
                                      ' --Graph Distance-Correlation-Line --MergeIndex 6')
                                      
    os.system(shared_params_st_mgcn + ' --City Chongqing --K 1 --L 1  '
                                      ' --Graph Distance-Correlation-Line --MergeIndex 12')
                