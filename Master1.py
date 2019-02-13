import os
# from Master0 import shared_params_gcn

shared_params_gcn = 'python -m Experiment.GCN_MultiGraph'   +\
                    ' --Train True'                        +\
                    ' --patience 50'              +\
                    ' --Epoch 10000 --BatchSize 64'        +\
                    ' --Device {} '

if __name__ == "__main__":

    print(shared_params_gcn)

    os.system(shared_params_gcn.format('1') + \
              '--City DC --Graph Distance-Correlation-Interaction --lr 1e-4 --K 1,1,1 --L 1,1,1 --CodeVersion lr1e4')

    os.system(shared_params_gcn.format('1') + \
              '--City DC --Graph Distance-Correlation-Interaction --lr 2e-4 --K 1,1,1 --L 1,1,1 --CodeVersion lr1e4')