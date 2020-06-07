
import numpy as np

from UCTB.model_unit import GraphBuilder
from UCTB.preprocess import Normalizer, SplitData

class GraphGenerator():
    '''
    Can use this class to build graph. 
    Adajacent matrix and lapalace matrix will be stored in self.AM and self.LM.

    Args:
    graph (str): Types of graphs used in neural methods. Graphs should be a subset of { ``'Correlation'``,
        ``'Distance'``, ``'Interaction'``, ``'Line'``, ``'Neighbor'``, ``'Transfer'`` } and concatenated by ``'-'``,
        and *dataset* should have data of selected graphs. Default: ``'Correlation'``
    threshold_distance (float): Used in building of distance graph. If distance of two nodes in meters is larger
        than ``threshold_distance``, the corresponding position of the distance graph will be 1 and otherwise
        0.the corresponding Default: 1000
    threshold_correlation (float): Used in building of correlation graph. If the Pearson correlation coefficient is
        larger than ``threshold_correlation``, the corresponding position of the correlation graph will be 1
        and otherwise 0. Default: 0
    threshold_interaction (float): Used in building of interatction graph. If in the latest 12 months, the number of
        times of interaction between two nodes is larger than ``threshold_interaction``, the corresponding position
        of the interaction graph will be 1 and otherwise 0. Default: 500
    '''

    def __init__(self,
                 graph,
                 dataset,
                 train_data,
                 traffic_data_index,
                 train_test_ratio,
                 threshold_distance=1000,
                 threshold_correlation=0,
                 threshold_interaction=500,**kwargs):
        self.AM = []
        self.LM = []
        self.threshold_distance = threshold_distance
        self.threshold_correlation = threshold_correlation
        self.threshold_interaction = threshold_interaction

        self.dataset = dataset
        self.train_data = train_data
        self.traffic_data_index = traffic_data_index
        self.train_test_ratio = train_test_ratio,
        self.daily_slots = 24 * 60 / self.dataset.time_fitness

        # build_graph
        for graph_name in graph.split('-'):
            AM, LM = self.build_graph(graph_name)
            if AM is not None:
                self.AM.append(AM)
            if LM is not None:
                self.LM.append(LM)
        
        self.AM = np.array(self.AM, dtype=np.float32)
        self.LM = np.array(self.LM, dtype=np.float32)
        # print (self.LM.shape[:])

    def build_graph(self, graph_name):
        AM, LM = None, None
        if graph_name.lower() == 'distance':
            lat_lng_list = np.array([[float(e1) for e1 in e[2:4]]
                                     for e in self.dataset.node_station_info])
            AM = GraphBuilder.distance_adjacent(lat_lng_list[self.traffic_data_index],
                                                threshold=float(self.threshold_distance))
            LM = GraphBuilder.adjacent_to_laplacian(AM)

        if graph_name.lower() == 'interaction':
            monthly_interaction = self.dataset.node_monthly_interaction[:, self.traffic_data_index, :][:, :,
                                                                                                       self.traffic_data_index]

            monthly_interaction, _ = SplitData.split_data(
                monthly_interaction, self.train_test_ratio)

            annually_interaction = np.sum(monthly_interaction[-12:], axis=0)
            annually_interaction = annually_interaction + annually_interaction.transpose()

            AM = GraphBuilder.interaction_adjacent(annually_interaction,
                                                   threshold=float(self.threshold_interaction))
            LM = GraphBuilder.adjacent_to_laplacian(AM)

        if graph_name.lower() == 'correlation':
            AM = GraphBuilder.correlation_adjacent(self.train_data[-30 * int(self.daily_slots):],
                                                   threshold=float(self.threshold_correlation))
            LM = GraphBuilder.adjacent_to_laplacian(AM)

        if graph_name.lower() == 'neighbor':
            LM = GraphBuilder.adjacent_to_laplacian(
                self.dataset.data.get('contribute_data').get('graph_neighbors'))

        if graph_name.lower() == 'line':
            LM = GraphBuilder.adjacent_to_laplacian(
                self.dataset.data.get('contribute_data').get('graph_lines'))
            LM = LM[self.traffic_data_index]
            LM = LM[:, self.traffic_data_index]

        if graph_name.lower() == 'transfer':
            LM = GraphBuilder.adjacent_to_laplacian(
                self.dataset.data.get('contribute_data').get('graph_transfer'))

        
        return AM, LM