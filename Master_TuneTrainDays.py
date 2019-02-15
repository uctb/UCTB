import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gcn = 'python -m Experiment.GCN_MultiGraph'   +\
                    ' --Train True'                        +\
                    ' --patience 50'              +\
                    ' --Epoch 10000 --BatchSize 64'        +\
                    ' --Device {} '

if __name__ == "__main__":

    print(shared_params_gcn)

    # os.system(shared_params_gcn.format('1') + \
    #           '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 '
    #           '--TrainDays All --lr 1e-3 '
    #           '--CodeVersion lr1e3')
    #
    # os.system(shared_params_gcn.format('1') + \
    #           '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 '
    #           '--TrainDays 365 --lr 5e-4 '
    #           '--CodeVersion TuneTrainDays365')
    #
    # os.system(shared_params_gcn.format('1') + \
    #           '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 '
    #           '--TrainDays 178 --lr 5e-4 '
    #           '--CodeVersion TuneTrainDays178')
    #
    # os.system(shared_params_gcn.format('1') + \
    #           '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 '
    #           '--TrainDays 89 --lr 5e-4 '
    #           '--CodeVersion TuneTrainDays89')

    os.system(shared_params_gcn.format('1') + \
              '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 '
              '--TrainDays 730 --lr 5e-4 '
              '--CodeVersion TuneTrainDays730')