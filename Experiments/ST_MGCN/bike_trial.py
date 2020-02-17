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
                         '--Device 1 ')

"""
V1
LSTMLayers : 3 => 4
K : 1 => 3
"""

if __name__ == "__main__":

    """
    Multiple Graphes
    """
    # os.system(shared_params_st_mgcn + ' --City NYC --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 3')
    os.system(shared_params_st_mgcn + ' --City NYC --Dataset Bike --K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 6')
    os.system(shared_params_st_mgcn + ' --City NYC --Dataset Bike--K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction --MergeIndex 12')

    # os.system(shared_params_st_mgcn + ' --City Chicago --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 3')
    # os.system(shared_params_st_mgcn + ' --City Chicago --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 6')
    # os.system(shared_params_st_mgcn + ' --City Chicago --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 12')
    # #
    # os.system(shared_params_st_mgcn + ' --City DC --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 3')
    # os.system(shared_params_st_mgcn + ' --City DC --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 6')
    # os.system(shared_params_st_mgcn + ' --City DC --Dataset Bike --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance-Correlation-Interaction --MergeIndex 12')