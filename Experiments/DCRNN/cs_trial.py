import os

import warnings
warnings.filterwarnings("ignore")

shared_params_dcrnn = ('python DCRNN.py '
                         '--Dataset ChargeStation '
                         '--CT 6 '
                         '--PT 0 '
                         '--TT 0 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 1 '
                         '--DataRange All '
                         '--TrainDays 365 '
                         '--TC 0.1 '
                         '--TD 1000 '
                         '--TI 500 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 5e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--Device 1 ')

if __name__ == "__main__":
    """
    Multiple Graphes
    """
    os.system(shared_params_dcrnn + ' --City Beijing --Dataset Chargestation --K 1 --L 1 '
                                      ' --Graph Distance --MergeIndex 1')

    os.system(shared_params_dcrnn + ' --City Beijing --Dataset Chargestation --K 1 --L 1 '
                                        ' --Graph Distance --MergeIndex 2')