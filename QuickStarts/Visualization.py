from UCTB.dataset import NodeTrafficLoader
#from UCTB.utils import st_map

from dateutil.parser import parse

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', with_lm=False)

data_loader.st_map()