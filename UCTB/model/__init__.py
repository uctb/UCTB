from .ModelObject import ModelObject

from .HM import HM
from .ARIMA import ARIMA

try:
    from .HMM import HMM
except ModuleNotFoundError:
    print('HMM not installed')

from .XGBoost import XGBoost
from .GBRT import GBRT

from .DeepST import DeepST
from .ST_ResNet import ST_ResNet

from .AMulti_GCLSTM import AMulti_GCLSTM

from .DCRNN import DCRNN

from .ST_MGCN import ST_MGCN