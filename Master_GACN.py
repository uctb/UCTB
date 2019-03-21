import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gacn = ('python -m Experiment.GACN --Train True --Dataset ChargeStation'
                      ' --City Beijing --Graph Correlation'
                      ' --Epoch 10000 --BatchSize 1 --lr 1e-4 --patience 50'
                      ' --GCLK 1 --GCLLayer 1 --GALLayer 4 --GALUnits 16 --GALHead 2 --DenseUnits 32'
                      ' --TC 0 --TD 1000 --TI 500'
                      ' --TrainDays All'
                      ' --Device 0')

if __name__ == "__main__":

    os.system(shared_params_gacn + ' --T 72 --CodeVersion V0')
    os.system(shared_params_gacn + ' --T 48 --CodeVersion V0')
    os.system(shared_params_gacn + ' --T 24 --CodeVersion V0')