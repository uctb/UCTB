import pandas as pd
from UCTB.preprocess.RegionGenerator import RegionGenerator
from UCTB.preprocess.dataset_helper import convert_uctb_data

# NYC
# initialize the configuration of the RegionGenerator
spatial_range = []
area_limit = 1

region_generator = RegionGenerator(spatial_range=spatial_range,area_limit=area_limit)

# regions are created in self.regions
region_generator.partition(method='grid')

service_record_filepath = 'trip_record.csv'
df = pd.read_csv(service_record_filepath)

# service records are binded in self.demand_matrix
region_generator.bind(df,method='location_based')

# cluster elements to acquire regions with better charateristics
regions,demand_matrix = region_generator.aggregate(cluster_method='node_swapping')


time_fitness = 5
time_range = ['2013-01-01','2014-01-01']

uctb_data = convert_uctb_data(regions,demand_matrix,time_fitness,time_range)






