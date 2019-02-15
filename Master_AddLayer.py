import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gcn = 'python -m Experiment.GCN_MultiGraph_AddLayer'   +\
                    ' --Train True'                        +\
                    ' --patience 50'              +\
                    ' --Epoch 10000 --BatchSize 64'        +\
                    ' --Device {} '

if __name__ == "__main__":


    os.system(shared_params_gcn.format('0') + \
              '--City DC --lr 5e-4 --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --GLL 3 --CodeVersion GLL3lr5e4')

    # os.system(shared_params_gcn.format('0') + \
    #           '--City DC --lr 5e-4 --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --GLL 2 --CodeVersion GLL2')