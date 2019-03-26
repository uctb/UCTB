import os

import warnings
warnings.filterwarnings("ignore")

shared_params_amulti_gclstm = ('python -m Experiment.AMulti_GCLSTM '
                               '--Dataset Subway '
                               '--City Shanghai '
                               '--T 6 '
                               
                               '--LSTMUnits 256 '
                               '--GALUnits 256 '
                               
                               '--GALHeads 2 '
                               '--DenseUnits 32 '
                               
                               '--DataRange All '
                               '--TrainDays All '
                               '--TC 0 '
                               '--TD 1000 '
                               '--TI 500 '
                               '--Epoch 20000 '
                               '--Train True '
                               '--lr 2e-4 '
                               '--patience 0.1 '
                               '--ESlength 300 '
                               '--BatchSize 128 '
                               '--Device 0 ')

if __name__ == "__main__":

    # LSTM
    # os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 0 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion ES100V1')

    # Single Graphs
    os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 1 --L 1 --GLL 1'
                                            ' --Group StableTest --CodeVersion 6')
    os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 1 --L 1 --GLL 1'
                                            ' --Group StableTest --CodeVersion 7')
    os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 1 --L 1 --GLL 1'
                                            ' --Group StableTest --CodeVersion 8')
    os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 1 --L 1 --GLL 1'
                                            ' --Group StableTest --CodeVersion 9')
    os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 1 --L 1 --GLL 1'
                                            ' --Group StableTest --CodeVersion 10')


    # os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 2 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    # os.system(shared_params_amulti_gclstm + ' --Graph Correlation --K 3 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    #
    # os.system(shared_params_amulti_gclstm + ' --Graph Distance --K 1 --L 1 --GLL 1'
    #                                         ' --Group SingleGraph --CodeVersion ES100V2')
    # os.system(shared_params_amulti_gclstm + ' --Graph Distance --K 2 --L 1 --GLL 1'
    #                                         ' --Group SingleGraph -CodeVersion GLL1')
    # os.system(shared_params_amulti_gclstm + ' --Graph Distance --K 3 --L 1 --GLL 1'
    #                                         ' --Group SingleGraph --CodeVersion GLL1')
    #
    # os.system(shared_params_amulti_gclstm + ' --Graph Neighbor --K 1 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion ES100V2')
    # os.system(shared_params_amulti_gclstm + ' --Graph Neighbor --K 2 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    # os.system(shared_params_amulti_gclstm + ' --Graph Neighbor --K 3 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    #
    # os.system(shared_params_amulti_gclstm + ' --Graph Line --K 1 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion ES100V2')
    # os.system(shared_params_amulti_gclstm + ' --Graph Line --K 2 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    # os.system(shared_params_amulti_gclstm + ' --Graph Line --K 3 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    #
    # os.system(shared_params_amulti_gclstm + ' --Graph Transfer --K 1 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    # os.system(shared_params_amulti_gclstm + ' --Graph Transfer --K 2 --L 1 --GLL 1'
    #                                         ' --Group Naive --CodeVersion GLL1')
    # os.system(shared_params_amulti_gclstm + ' --Graph Transfer --K 3 --L 1 --GLL 1'
    #                                         ' --Group Naive --GroupCodeVersion GLL1')