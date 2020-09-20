import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN_Obj.py '
                         '--Dataset ChargeStation '
                         '--CT 6 '
                         '--PT 7 '
                         '--TT 4 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
                         '--DataRange All '
                         '--TrainDays All '
                         '--TC 0.1 '
                         '--TD 1000 '
                         '--TI 500 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 5e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--MergeWay max '
                         '--Device 1 ')

if __name__ == "__main__":
    
    os.system(shared_params_st_mgcn + ' --City Beijing --K 1 --L 1  '
                                      ' --Graph Distance-Correlation --MergeIndex 1')

    os.system(shared_params_st_mgcn + ' --City Beijing --K 1 --L 1  '
                                      ' --Graph Distance-Correlation --MergeIndex 2')