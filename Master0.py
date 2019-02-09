import os

shared_params_gcn = 'python -m Experiment.GCN_MultiGraph'   +\
                    ' --Train True'                        +\
                    ' --lr 5e-5 --patience 50'              +\
                    ' --Epoch 10000 --BatchSize 64'        +\
                    ' --Device {} '

if __name__ == "__main__":

    print(shared_params_gcn)

    # os.system(shared_params_gcn.format('0') + '--City NYC --Graph Distance --K 0 --L 1 --CodeVersion V0')
    # os.system(shared_params_gcn.format('0') + '--City NYC --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City NYC --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City NYC --Graph Correlation --K 1 --L 1 --CodeVersion V0')
    # os.system(shared_params_gcn.format('0') +\
    #           '--City NYC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion V0')

    os.system(shared_params_gcn.format('0') + '--City DC --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City DC --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City DC --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City DC --Graph Correlation --K 1 --L 1 --CodeVersion V0')
    # os.system(shared_params_gcn.format('0') + \
    #           '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion V0')

    os.system(shared_params_gcn.format('0') + '--City Chicago --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City Chicago --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City Chicago --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('0') + '--City Chicago --Graph Correlation --K 1 --L 1 --CodeVersion V0')
    # os.system(shared_params_gcn.format('0') + \
    #           '--City Chicago --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion V0')

    