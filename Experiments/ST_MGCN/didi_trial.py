import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN_Obj.py '
                         '--Dataset DiDi '
                         '--CT 6 '
                         '--PT 7 '
                         '--TT 4 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
                         '--DataRange All '
                         '--TrainDays 365 '
                         '--TC 0.65 '
                         '--TD 7500 '
                         '--TI 30 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 1e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--Device 1 ')

if __name__ == "__main__":

    """
    Multiple Graphes
    """
    os.system(shared_params_st_mgcn + ' --City Chengdu --Group Debug --K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction')

    os.system(shared_params_st_mgcn + ' --City Xian --Group Chicago --K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction')