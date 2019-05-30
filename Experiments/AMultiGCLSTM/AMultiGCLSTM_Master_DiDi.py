import os

import warnings
warnings.filterwarnings("ignore")

shared_params_amulti_gclstm = ('python AMulti_GCLSTM.py '
                               '--Dataset DiDi '
                               '--T 6 '
                               '--GLL 1 '
                               '--LSTMUnits 64 '
                               '--GALUnits 64 '
                               '--GALHeads 2 '
                               '--DenseUnits 32 '
                               '--DataRange All '
                               '--TrainDays All '
                               '--TC 0 '
                               '--TD 1000 '
                               '--TI 500 '
                               '--Epoch 5000 '
                               '--Train False '
                               '--lr 5e-4 '
                               '--patience 0.1 '
                               '--ESlength 50 '
                               '--BatchSize 64 '
                               '--Device 0 ')

if __name__ == "__main__":

    """
    Single Graph
    """
    os.system(shared_params_amulti_gclstm + ' --City Chengdu --Group Chengdu --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Chengdu --Group Chengdu --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Chengdu --Group Chengdu --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Chengdu --Group Chengdu --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    os.system(shared_params_amulti_gclstm + ' --City Xian --Group Xian --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Xian --Group Xian --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Xian --Group Xian --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Xian --Group Xian --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    """
    Multiple Graphes
    """
    os.system(shared_params_amulti_gclstm + ' --City Chengdu --Group Chengdu --K 1 --L 1 --CodeVersion V0'
                                            ' --Graph Distance-Interaction-Correlation')

    os.system(shared_params_amulti_gclstm + ' --City Xian --Group Xian --K 1 --L 1 --CodeVersion V0'
                                            ' --Graph Distance-Interaction-Correlation')