import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN_Obj.py '
                         '--Dataset Bike '
                         '--CT 6 '
                         '--PT 7 '
                         '--TT 4 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
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
                         '--BatchSize 16 '
                         '--MergeWay sum '
                         '--Device 1 '
                         '')

"""
V1
LSTMLayers : 3 => 4
K : 1 => 3
"""
if __name__ == "__main__":

    """
    Multiple Graphs
    """
    # NYC
    os.system(shared_params_st_mgcn + ' --City NYC --K 1 --L 1 --DataRange 0.125 --TrainDays 60'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 1')
    os.system(shared_params_st_mgcn + ' --City NYC --K 1 --L 1 --DataRange 0.25 --TrainDays 91'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 3')
    os.system(shared_params_st_mgcn + ' --City NYC --K 1 --L 1 --DataRange 0.5 --TrainDays 183'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 6')
    os.system(shared_params_st_mgcn + ' --City NYC --K 1 --L 1 '
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 12')
    
    # Chicago 
    os.system(shared_params_st_mgcn + ' --City Chicago --K 1 --L 1 --DataRange 0.125 --TrainDays 60'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 1')
    os.system(shared_params_st_mgcn + ' --City Chicago --K 1 --L 1 --DataRange 0.25 --TrainDays 91'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 3')
    os.system(shared_params_st_mgcn + ' --City Chicago --K 1 --L 1 --DataRange 0.5 --TrainDays 183'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 6')
    os.system(shared_params_st_mgcn + ' --City Chicago --K 1 --L 1 '
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 12')
    
    # DC
    os.system(shared_params_st_mgcn + ' --City DC --K 1 --L 1 --DataRange 0.125 --TrainDays 60'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 1')
    os.system(shared_params_st_mgcn + ' --City DC --K 1 --L 1 --DataRange 0.25 --TrainDays 91'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 3')
    os.system(shared_params_st_mgcn + ' --City DC --K 1 --L 1 --DataRange 0.5 --TrainDays 183'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 6')
    os.system(shared_params_st_mgcn + ' --City DC --K 1 --L 1 '
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 12')
