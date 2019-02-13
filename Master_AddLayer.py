import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gcn = 'python -m Experiment.GCN_MultiGraph2'   +\
                    ' --Train True'                        +\
                    ' --lr 5e-5 --patience 50'              +\
                    ' --Epoch 10000 --BatchSize 64'        +\
                    ' --Device {} '

if __name__ == "__main__":


    os.system(shared_params_gcn.format('0') + \
              '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion AddLayer')