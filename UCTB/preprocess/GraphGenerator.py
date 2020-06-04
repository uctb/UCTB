import os
import nni
import yaml
import argparse
import GPUtil
import numpy as np
from UCTB.dataset import DataSet

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import STMeta
from UCTB.evaluation import metric
from UCTB.preprocess.time_utils import is_work_day_china, is_work_day_america
from UCTB.utils.sendInfo import senInfo
from UCTB.model_unit import GraphBuilder
from UCTB.preprocess import Normalizer, SplitData

class GraphGenerator():
    '''
    the class can be move to UTCB/preprocess dir
    If users want to extend it, they just need to inherit or modify this class.
    '''

    def __init__(self,
                 graph,
                 node_data,
                 train_data,
                 traffic_data_index,
                 train_test_ratio,
                 threshold_distance=1000,
                 threshold_correlation=0,
                 threshold_interaction=500,
                 ):
        self.AM = []
        self.LM = []
        self.threshold_distance = threshold_distance
        self.threshold_correlation = threshold_correlation
        self.threshold_interaction = threshold_interaction

        self.dataset = node_data
        self.train_data = train_data
        self.traffic_data_index = traffic_data_index
        self.train_test_ratio = train_test_ratio
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