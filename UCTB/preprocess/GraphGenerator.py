import numpy as np
import tensorflow as tf
import heapq

# from UCTB.preprocess import Normalizer, SplitData

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr
from scipy.sparse.linalg import eigs

class GraphGenerator():
    '''
    This class is used to build graphs. 
    Adajacent matrix and lapalace matrix will be stored in self.AM and self.LM.

    Args:
        data_loader(NodeTrafficLoader): data_loader object.
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

    Attributes:
        AM (array): Adajacent matrices of graphs.
        LM (array): Laplacian matrices of graphs.
    '''

    def __init__(self,
                 data_loader,
                 graph="Correlation",
                 threshold_distance=1000,
                 threshold_correlation=0,
                 threshold_interaction=500, **kwargs):
        self.AM = []
        self.LM = []
        self.threshold_distance = threshold_distance
        self.threshold_correlation = threshold_correlation
        self.threshold_interaction = threshold_interaction

        self.dataset = data_loader.dataset
        self.train_data = data_loader.train_data
        self.traffic_data_index = data_loader.traffic_data_index
        self.train_test_ratio = data_loader.train_test_ratio
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
            AM = self.distance_adjacent(lat_lng_list[self.traffic_data_index],
                                                threshold=float(self.threshold_distance))
            LM = self.adjacent_to_laplacian(AM)

        if graph_name.lower() == 'interaction':
            monthly_interaction = self.dataset.node_monthly_interaction[:, self.traffic_data_index, :][:, :,
                                                                                                       self.traffic_data_index]

            monthly_interaction, _ = SplitData.split_data(
                monthly_interaction, self.train_test_ratio)

            annually_interaction = np.sum(monthly_interaction[-12:], axis=0)
            annually_interaction = annually_interaction + annually_interaction.transpose()

            AM = self.interaction_adjacent(annually_interaction,
                                                   threshold=float(self.threshold_interaction))
            LM = self.adjacent_to_laplacian(AM)

        if graph_name.lower() == 'correlation':
            AM = self.correlation_adjacent(self.train_data[-30 * int(self.daily_slots):],
                                                   threshold=float(self.threshold_correlation))
            LM = self.adjacent_to_laplacian(AM)

        if graph_name.lower() == 'neighbor':
            LM = self.adjacent_to_laplacian(
                self.dataset.data.get('contribute_data').get('graph_neighbors'))

        if graph_name.lower() == 'line':
            LM = self.adjacent_to_laplacian(
                self.dataset.data.get('contribute_data').get('graph_lines'))
            LM = LM[self.traffic_data_index]
            LM = LM[:, self.traffic_data_index]

        if graph_name.lower() == 'transfer':
            LM = self.adjacent_to_laplacian(
                self.dataset.data.get('contribute_data').get('graph_transfer'))
        return AM, LM

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371

        return c * r * 1000

    @staticmethod
    def correlation_adjacent(traffic_data, threshold):
        '''
        Calculate correlation graph based on pearson coefficient.

        Args:
            traffic_data(ndarray): numpy array with shape [sequence_length, num_node].
            threshold(float): float between [-1, 1], nodes with Pearson Correlation coefficient
                larger than this threshold will be linked together.
        '''
        adjacent_matrix = np.zeros([traffic_data.shape[1], traffic_data.shape[1]])
        for i in range(traffic_data.shape[1]):
            for j in range(traffic_data.shape[1]):
                r, p_value = pearsonr(traffic_data[:, i], traffic_data[:, j])
                adjacent_matrix[i, j] = 0 if np.isnan(r) else r
        adjacent_matrix = (adjacent_matrix >= threshold).astype(np.float32)
        return adjacent_matrix

    def distance_adjacent(self, lat_lng_list, threshold):
        '''
        Calculate distance graph based on geographic distance.

        Args:
            lat_lng_list(list): A list of geographic locations. The format of each element
                    in the list is [latitude, longitude].
            threshold(float): (meters) nodes with geographic distacne smaller than this 
                threshold will be linked together.
        '''
        adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
        for i in range(len(lat_lng_list)):
            for j in range(len(lat_lng_list)):
                adjacent_matrix[i][j] = self.haversine(lat_lng_list[i][0], lat_lng_list[i][1],
                                                                lat_lng_list[j][0], lat_lng_list[j][1])
        adjacent_matrix = (adjacent_matrix <= threshold).astype(np.float32)
        return adjacent_matrix

    @staticmethod
    def interaction_adjacent(interaction_matrix, threshold):
        '''
        Binarize interaction_matrix based on threshold.

        Args:

            interaction_matrix(ndarray): with shape [num_node, num_node], where each 
                element represents the number of interactions during a certain time,
                    e.g. 6 monthes, between the corresponding nodes.
            threshold(float or int): nodes with number of interactions between them
                    greater than this threshold will be linked together.
        '''
        return (interaction_matrix >= threshold).astype(np.float32)      

    @staticmethod
    def adjacent_to_laplacian(adjacent_matrix):
        '''
        Turn adjacent_matrix into Laplace matrix.
        '''
        adjacent_matrix -= np.diag(np.diag(adjacent_matrix))
        diagonal_matrix = np.diag(np.sum(adjacent_matrix, axis=0) ** -0.5)
        diagonal_matrix[np.isinf(diagonal_matrix)] = 0
        laplacian_matrix = np.eye(len(adjacent_matrix)) - np.dot(np.dot(diagonal_matrix, adjacent_matrix),
                                                                    diagonal_matrix)
        laplacian_matrix = 2 * laplacian_matrix / np.max(laplacian_matrix) - np.eye(len(adjacent_matrix))
        return laplacian_matrix



def scaled_Laplacian_ASTGCN(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W(np.ndarray): shape is (num_node, num_node).

    Returns
    ----------
    scaled_Laplacian_ASTGCN: np.ndarray, shape (num_node, num_node)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def scaled_laplacian_STGCN(W):
    '''
    Normalized graph Laplacian function.

    Args:
        W(np.ndarray): [num_node, num_node], weighted adjacency matrix of G.
    :return: Scaled laplacian matrix.
    :type: np.matrix, [num_node, num_node].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))
