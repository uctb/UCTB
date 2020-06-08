from UCTB.preprocess.GraphGenerator import GraphGenerator
import numpy as np

class CusGraph(GraphGenerator):# Init NodeTrafficLoader
    def __init__(self, graph, dataset, train_data, traffic_data_index, train_test_ratio, threshold_distance, threshold_correlation, threshold_interaction, threshold_neighbour=20, with_lm=True):
        super(CusGraph, self).__init__(graph, dataset, train_data, traffic_data_index, train_test_ratio, threshold_distance, threshold_correlation, threshold_interaction, threshold_neighbour=20, with_lm=True) # [!INFO] Init NodeTrafficLoader

        if with_lm:# Vitial if use lm as the imput
            for graph_name in graph.split('-'):# As the basic graph is implemented in NodeTrafficLoader, you only need to implement your own graph function
                ifchange = False# Whether change

                # Sample of topk

                if graph_name.lower() == 'topk':
                    ifchange = True
                    lat_lng_list = np.array([[float(e1) for e1 in e[2:4]]
                                            for e in self.dataset.node_station_info])
                    # Handling
                    AM = neighbour_adjacent(lat_lng_list[self.traffic_data_index],
                                                        threshold=int(threshold_neighbour))
                    LM = adjacent_to_laplacian(AM)
                
                # Sample for directly use
                # if graph_name.lower() == 'direct':
                #     ifchange = True
                #     LM = self.dataset.data.get('primary_dir').get('secondary_dir')

                if ifchange:# if analyze with new graph
                    # Combining
                    if self.AM.shape[0] == 0:# Make AM
                        self.AM = np.array([AM], dtype=np.float32)
                    else:
                        self.AM = np.vstack((self.AM, (AM[np.newaxis, :])))

                    if self.LM.shape[0] == 0:# Make LM
                        self.LM = np.array([LM], dtype=np.float32)
                    else:
                        self.LM = np.vstack((self.LM, (LM[np.newaxis, :])))

def neighbour_adjacent(lat_lng_list, threshold):
    adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
    for i in range(len(lat_lng_list)):
        for j in range(len(lat_lng_list)):
            adjacent_matrix[i][j] = haversine(lat_lng_list[i][0], lat_lng_list[i][1],lat_lng_list[j][0], lat_lng_list[j][1])
    dis_matrix = adjacent_matrix.astype(np.float32)
    for i in range(len(dis_matrix)):
        ind = heapq.nlargest(threshold, range(len(dis_matrix[i])), dis_matrix[i].take)
        dis_matrix[i] = np.array([0 for _ in range(len(dis_matrix[i]))])
        dis_matrix[i][ind] = 1
    adjacent_matrix = (adjacent_matrix == 1).astype(np.float32)
    return adjacent_matrix      