import os
from local_path import *

NYC_Prediction = [e for e in os.listdir(data_dir) if e.endswith('.npy') and 'NYC' in e and 'K1L1' in e and 'V0' in e]
Chicago_Prediction = [e for e in os.listdir(data_dir) if e.endswith('.npy') and 'Chicago' in e and 'K1L1' in e and 'V0' in e]
DC_Prediction = [e for e in os.listdir(data_dir) if e.endswith('.npy') and 'DC' in e and 'K1L1' in e and 'V0' in e]

print(DC_Prediction)

