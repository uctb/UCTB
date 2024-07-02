import logging
import numpy as np
def grid_partition():
    # To be implemented
    pass

def hexagon_partition():
    # To be implemented
    pass

def roadnetwork_partition():
    # To be implemented
    pass

def location_bind():
    # To be implemented
    pass

def async_fluid():
    # To be implemented
    pass

def node_swapping():
    # To be implemented
    pass

class RegionGenerator():
    '''
    This class is used to generate regions and create demand matrix.
    Regions will be stored into self.regions and demand matrix will be stored into self.demand_matrix
    This class should be instantiated before NodeTrafficLoader is instantiated.
    Args:
        spatial_range: list([lat_min,lat_max,lon_min,lon_max])
        area_limit: int (All regions should be smaller than)


    '''
    partition_func_dict = {
        'grid':grid_partition,
        'hexagon':hexagon_partition,
        'road_network':roadnetwork_partition
    }
    bind_func_dict = {
        'location':location_bind
    }
    cluster_func_dict = {
        'async_fluid':async_fluid,
        'node_swapping':node_swapping
    }
    def __init__(self,spatial_range,area_limit) -> None:
        self.lat_min = spatial_range[0]
        self.lat_max = spatial_range[1]
        self.lon_min = spatial_range[2]
        self.lon_max = spatial_range[3]
        self.area_limit = area_limit


    def partition(self,method,**params) -> any:
        if method not in self.partition_func_dict:
            logging.error(f"Unsupported method of partition: {method}. Skipping.")
        else:
            self.regions = self.partition_func_dict[method](**params)

    def bind(self,df,method,**params) -> any:
        if method not in self.bind_func_dict:
            logging.error(f"Unsupported method of bind: {method}. Skipping.")
        else:
            self.demand_matrix = self.bind_func_dict[method](**params)
        pass

    def aggregate(self,cluster_method,merge_way='sum',**params) -> any:
        if cluster_method not in self.partition_func_dict:
            logging.error(f"Unsupported method of aggregation: {cluster_method}. Skipping.")
        else:
            regions2clusters,_ = self.cluster_func_dict[cluster_method](self.regions,**params)
        num_clusters = max(regions2clusters)
        new_demand_matrix = np.zeros([self.demand_matrix.shape[0],num_clusters])
        for i in range(num_clusters):
            
            if merge_way == 'sum':
                new_demand_matrix[:,i] = np.sum(self.demand_matrix[:,np.nonzero(regions2clusters==i)[0]])
            elif merge_way == 'average':
                new_demand_matrix[:,i] = np.average(self.demand_matrix[:,np.nonzero(regions2clusters==i)[0]])
            else:
                raise KeyError('not implemented')
        new_regions = None # aggregate regions in to new regions (should be a single geometry but not collections of geometry)
        return new_regions,new_demand_matrix

        

