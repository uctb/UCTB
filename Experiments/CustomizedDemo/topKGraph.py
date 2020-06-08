import heapq
import numpy as np
from UCTB.preprocess.GraphGenerator import GraphGenerator


class topKGraph(GraphGenerator):  # Init NodeTrafficLoader

    def __init__(self,**kwargs):

        super(topKGraph, self).__init__(**kwargs)
        
        for graph_name in kwargs['graph'].split('-'):
            if graph_name.lower() == 'topk':
                lat_lng_list = np.array([[float(e1) for e1 in e[2:4]]
                                         for e in self.dataset.node_station_info])
                # Handling
                AM = self.neighbour_adjacent(lat_lng_list[self.traffic_data_index],
                                        threshold=int(kwargs['threshold_neighbour']))
                LM = self.adjacent_to_laplacian(AM)

                if self.AM.shape[0] == 0:  # Make AM
                    self.AM = np.array([AM], dtype=np.float32)
                else:
                    self.AM = np.vstack((self.AM, (AM[np.newaxis, :])))

                if self.LM.shape[0] == 0:  # Make LM
                    self.LM = np.array([LM], dtype=np.float32)
                else:
                    self.LM = np.vstack((self.LM, (LM[np.newaxis, :])))

    def neighbour_adjacent(self, lat_lng_list, threshold):
        adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
        for i in range(len(lat_lng_list)):
            for j in range(len(lat_lng_list)):
                adjacent_matrix[i][j] = self.haversine(
                    lat_lng_list[i][0], lat_lng_list[i][1], lat_lng_list[j][0], lat_lng_list[j][1])
        dis_matrix = adjacent_matrix.astype(np.float32)

        for i in range(len(dis_matrix)):
            ind = heapq.nlargest(threshold, range(len(dis_matrix[i])), dis_matrix[i].take)
            dis_matrix[i] = np.array([0 for _ in range(len(dis_matrix[i]))])
            dis_matrix[i][ind] = 1
        adjacent_matrix = (adjacent_matrix == 1).astype(np.float32)
        return adjacent_matrix
