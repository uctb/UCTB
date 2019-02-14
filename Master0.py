import os

import warnings
warnings.filterwarnings("ignore")

shared_params_gcn = 'python -m Experiment.GCN_MultiGraph'   +\
                    ' --Train False'                        +\
                    ' --lr 5e-5 --patience 50'              +\
                    ' --Epoch 10000 --BatchSize 64'        +\
                    ' --Device {} '

if __name__ == "__main__":

    # MGCN

    # print(shared_params_gcn)
    # os.system(shared_params_gcn.format('0') + '--City NYC --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City NYC --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City NYC --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City NYC --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    # os.system(shared_params_gcn.format('0') + '--City DC --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City DC --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City DC --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City DC --Graph Correlation --K 1 --L 1 --CodeVersion V0')

    # os.system(shared_params_gcn.format('0') + '--City Chicago --Graph Distance --K 0 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City Chicago --Graph Distance --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City Chicago --Graph Interaction --K 1 --L 1 --CodeVersion V0')
    os.system(shared_params_gcn.format('2') + '--City Chicago --Graph Correlation --K 1 --L 1 --CodeVersion V0')


    # XGBoost
    # os.system("python -m Experiment.XGBOOST --City NYC --CodeVersion V0")
    # os.system("python -m Experiment.XGBOOST --City DC --CodeVersion V0")
    # os.system("python -m Experiment.XGBOOST --City Chicago --CodeVersion V0")

    # HMM
    # os.system("python -m Experiment.HMM_Obj --City NYC --CodeVersion V0")
    # os.system("python -m Experiment.HMM_Obj --City DC --CodeVersion V0")
    # os.system("python -m Experiment.HMM_Obj --City Chicago --CodeVersion V0")

    # HM
    # os.system("python -m Experiment.HM --City NYC --CodeVersion V0")
    # os.system("python -m Experiment.HM --City DC --CodeVersion V0")
    # os.system("python -m Experiment.HM --City Chicago --CodeVersion V0")