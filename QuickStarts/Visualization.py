from UCTB.dataset import NodeTrafficLoader
from UCTB.utils import st_map

from dateutil.parser import parse

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', with_lm=False)

sorted_items = sorted(data_loader.dataset.node_station_info.items(), key=lambda x: parse(x[1][0]), reverse=False)

st_map(lat=[e[1][1] for e in sorted_items],
       lng=[e[1][2] for e in sorted_items],
       build_order=[e for e in range(len(sorted_items))],
       meta_info=[e[1][3] for e in sorted_items],
       file_name='Bike_NYC.html')