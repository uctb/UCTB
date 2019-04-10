import os

import warnings
warnings.filterwarnings("ignore")

shared_params_amulti_gclstm = ('python AMulti_GCLSTM.py '
                               '--Dataset Bike '
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
                               '--Train True '
                               '--lr 5e-4 '
                               '--patience 0.1 '
                               '--ESlength 50 '
                               '--BatchSize 64 '
                               '--Device 0 ')

if __name__ == "__main__":

    """
    Single Graph
    """
    os.system(shared_params_amulti_gclstm + ' --City NYC --Group NYC --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City NYC --Group NYC --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City NYC --Group NYC --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City NYC --Group NYC --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    os.system(shared_params_amulti_gclstm + ' --City Chicago --Group Chicago --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Chicago --Group Chicago --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Chicago --Group Chicago --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City Chicago --Group Chicago --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    os.system(shared_params_amulti_gclstm + ' --City DC --Group DC --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City DC --Group DC --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City DC --Group DC --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_amulti_gclstm + ' --City DC --Group DC --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    """
    Multiple Graphes
    """
    os.system(shared_params_amulti_gclstm + ' --City NYC --Group NYC --K 1 --L 1 --CodeVersion V0'
                                            ' --Graph Distance-Interaction-Correlation')

    os.system(shared_params_amulti_gclstm + ' --City Chicago --Group Chicago --K 1 --L 1 --CodeVersion V0'
                                            ' --Graph Distance-Interaction-Correlation')

    os.system(shared_params_amulti_gclstm + ' --City DC --Group DC --K 1 --L 1 --CodeVersion V0'
                                            ' --Graph Distance-Interaction-Correlation')