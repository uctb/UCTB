import os
from Master0 import shared_params_gcn

if __name__ == "__main__":

    print(shared_params_gcn)
    
    os.system(shared_params_gcn.format('1') +\
              '--City NYC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion V0')

    os.system(shared_params_gcn.format('1') + \
              '--City DC --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion V0')

    os.system(shared_params_gcn.format('1') + \
              '--City Chicago --Graph Distance-Correlation-Interaction --K 1,1,1 --L 1,1,1 --CodeVersion V0')