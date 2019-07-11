import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gacn = ('python V3_GACN.py '
                      '--K 1 '
                      '--L 1 '
                      '--Graph Correlation '
                      '--DenseUnits 32 '
                      '--DataRange All '
                      '--TrainDays All '
                      '--TC 0 '
                      '--TD 1000 '
                      '--TI 500 '
                      '--Epoch 10000 '
                      '--Train True '
                      '--ESlength 50 '
                      '--patience 0.1 '
                      '--BatchSize 16 '
                      '--Device 1')


if __name__ == "__main__":

    # os.system(shared_params_gacn + ' --Dataset Bike --City Chicago --Group Chicago'
    #                                ' --lr 5e-5 --T 6 --GALLayers 4 --GALHeads 2 --GALUnits 32 --CodeVersion T6')

    os.system(shared_params_gacn + ' --Dataset Bike --City Chicago --Group Chicago'
                                   ' --lr 1e-4 --T 12 --GALLayers 4 --GALHeads 2 --GALUnits 32 --CodeVersion T12')