import os

import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN.py '
                         '--Dataset Bike '
                         '--T 6 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
                         '--DataRange All '
                         '--TrainDays All '
                         '--TC 0 '
                         '--TD 1000 '
                         '--TI 500 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 5e-4 '
                         '--patience 0.1 '
                         '--ESlength 50 '
                         '--BatchSize 64 '
                         '--Device 1 ')

if __name__ == "__main__":

    """
    Single Graph
    """
    # os.system(shared_params_st_mgcn + ' --City NYC --Group NYC --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance')
    # os.system(shared_params_st_mgcn + ' --City NYC --Group NYC --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Correlation')
    # os.system(shared_params_st_mgcn + ' --City NYC --Group NYC --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Interaction')
    #
    # os.system(shared_params_st_mgcn + ' --City Chicago --Group Chicago --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance')
    # os.system(shared_params_st_mgcn + ' --City Chicago --Group Chicago --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Correlation')
    # os.system(shared_params_st_mgcn + ' --City Chicago --Group Chicago --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Interaction')
    #
    # os.system(shared_params_st_mgcn + ' --City DC --Group DC --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Distance')
    # os.system(shared_params_st_mgcn + ' --City DC --Group DC --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Correlation')
    # os.system(shared_params_st_mgcn + ' --City DC --Group DC --K 1 --L 1 --CodeVersion V0'
    #                                   ' --Graph Interaction')

    """
    Multiple Graphes
    """
    os.system(shared_params_st_mgcn + ' --City NYC --Group NYC --K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction')

    os.system(shared_params_st_mgcn + ' --City Chicago --Group Chicago --K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction')

    os.system(shared_params_st_mgcn + ' --City DC --Group DC --K 1 --L 1 --CodeVersion V0'
                                      ' --Graph Distance-Correlation-Interaction')