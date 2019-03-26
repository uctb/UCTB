import os

import warnings
warnings.filterwarnings("ignore")

shared_params_amulti_gclstm = ('python -m Experiment.AMulti_GCLSTM '
                               '--Dataset Bike '
                               # '--City NYC'
                               # '--T 6 '
                               # '--K 1 '
                               # '--L 1 '
                               # '--Graph Correlation '
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
                               '--Device 1 '
                               '--Group BikeBasic')

if __name__ == "__main__":

    os.system(shared_params_amulti_gclstm + ' --T 6 --City NYC --Graph Distance --K 0 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City NYC --Graph Distance --K 3 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City NYC --Graph Interaction --K 3 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City NYC --Graph Correlation --K 3 --L 1 --CodeVersion GLL1')

    os.system(shared_params_amulti_gclstm + ' --T 6 --City DC --Graph Distance --K 0 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City DC --Graph Distance --K 3 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City DC --Graph Interaction --K 3 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City DC --Graph Correlation --K 3 --L 1 --CodeVersion GLL1')

    os.system(shared_params_amulti_gclstm + ' --T 6 --City Chicago --Graph Distance --K 0 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City Chicago --Graph Distance --K 3 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City Chicago --Graph Interaction --K 3 --L 1 --CodeVersion GLL1')
    os.system(shared_params_amulti_gclstm + ' --T 6 --City Chicago --Graph Correlation --K 3 --L 1 --CodeVersion GLL1')

    # os.system(shared_params_amulti_gclstm + ' --T 12 --City NYC --Graph Distance --K 1 --L 1 --CodeVersion T12')
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City NYC --Graph Interaction --K 1 --L 1 --CodeVersion T12')
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City NYC --Graph Correlation --K 1 --L 1 --CodeVersion T12')
    #
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City DC --Graph Distance --K 1 --L 1 --CodeVersion T12')
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City DC --Graph Interaction --K 1 --L 1 --CodeVersion T12')
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City DC --Graph Correlation --K 1 --L 1 --CodeVersion T12')
    #
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City Chicago --Graph Distance --K 1 --L 1 --CodeVersion T12')
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City Chicago --Graph Interaction --K 1 --L 1 --CodeVersion T12')
    # os.system(shared_params_amulti_gclstm + ' --T 12 --City Chicago --Graph Correlation --K 1 --L 1 --CodeVersion T12')
