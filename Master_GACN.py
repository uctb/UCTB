import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gacn = ('python -m Experiment.GACN_ChargeStation --Train True '
                      ' --City Beijing --Graph Correlation'
                      ' --Epoch 10000 --BatchSize 64 --lr 1e-3 --patience 50'
                      ' --GCLK 1 --GCLLayer 1 --GALLayer 4 --GALUnits 32 --GALHead 2 --DenseUnits 32'
                      ' --TC 0 --TD 1000 --TI 500'
                      ' --TrainDays All'
                      ' --Device 1')

if __name__ == "__main__":

    os.system(shared_params_gacn + ' --T 12 --CodeVersion V0')