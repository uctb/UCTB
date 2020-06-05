import numpy as np
import tensorflow as tf
import heapq

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr


class GraphBuilder(object):
    '''
    This class provides static methods for transforming raw data into various graphs.
    eg: correlation, distance, interaction graph.
    '''
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

    @staticmethod
    def distance_adjacent(lat_lng_list, threshold):
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
                adjacent_matrix[i][j] = GraphBuilder.haversine(lat_lng_list[i][0], lat_lng_list[i][1],
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
    def neighbour_adjacent(lat_lng_list, threshold):
        adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
        for i in range(len(lat_lng_list)):
            for j in range(len(lat_lng_list)):
                adjacent_matrix[i][j] = GraphBuilder.haversine(lat_lng_list[i][0], lat_lng_list[i][1],lat_lng_list[j][0], lat_lng_list[j][1])
        dis_matrix = adjacent_matrix.astype(np.float32)
        for i in range(len(dis_matrix)):
            ind = heapq.nlargest(threshold, range(len(dis_matrix[i])), dis_matrix[i].take)
            dis_matrix[i] = np.array([0 for _ in range(len(dis_matrix[i]))])
            dis_matrix[i][ind] = 1
        adjacent_matrix = (adjacent_matrix == 1).astype(np.float32)
        return adjacent_matrix            

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


# Graph Attention Layer
class GAL(object):
    '''
    This class provides static methods for adding Graph Attention Layer.
    '''
    @staticmethod
    def attention_merge_weight(inputs, units, num_head, activation=tf.nn.leaky_relu):
        inputs_shape = inputs.get_shape().with_rank(3)

        num_node = inputs_shape[-2].value
        num_feature = inputs_shape[-1].value

        W = tf.Variable(tf.random_normal([num_feature, units * num_head]))

        # linear transform
        l_t = tf.matmul(tf.reshape(inputs, [-1, num_feature]), W)
        l_t = tf.reshape(l_t, [-1, num_node, num_head, units])

        # compute attention
        a = tf.Variable(tf.random_normal([units * 2, num_head]))

        e = []
        for j in range(1, num_node):
            multi_head_result = []
            for k in range(num_head):
                multi_head_result.append(tf.matmul(tf.concat([l_t[:, 0, k, :], l_t[:, j, k, :]], axis=-1), a[:, k:k+1]))
            e.append(tf.reshape(tf.concat(multi_head_result, axis=-1), [-1, num_head, 1]))

        e = activation(tf.reshape(tf.concat(e, axis=-1), [-1, num_head, num_node-1]))

        alpha = tf.reduce_mean(tf.nn.softmax(e, axis=-1), axis=1, keepdims=True)

        return alpha

    @staticmethod
    def add_ga_layer_matrix(inputs, units, num_head, activation=tf.nn.tanh):
        '''
        This method use Multi-head attention technique to add Graph Attention Layer.

        Args:
            input(ndarray): The set of node features data, with shape [batch, num_node, num_featuer].
            unit(int): The number of merge_gal_units used in GAL.
            num_head(int): The number of multi-head used in GAL.
            activation(function): activation function. default:tf.nn.tanh.
        Returns:
            alpha: The weight matrix after softmax function.
            gc_output: The final GAL aggregated feature representation from input feature.
        '''
        inputs_shape = inputs.get_shape().with_rank(3)

        num_node = inputs_shape[-2].value
        num_feature = inputs_shape[-1].value

        W = tf.Variable(tf.random_normal([num_feature, units * num_head], dtype=tf.float32))

        # linear transform
        l_t = tf.matmul(tf.reshape(inputs, [-1, num_feature]), W)
        l_t = tf.reshape(l_t, [-1, num_node, num_head, units])

        a = tf.Variable(tf.random_normal([units * 2, num_head], dtype=tf.float32))

        e_multi_head = []

        for head_index in range(num_head):

            l_t_i = l_t[:, :, head_index, :]
            a_i = a[:, head_index:head_index+1]

            l_t_i_0 = tf.gather(l_t_i, indices=np.array([e for e in range(num_node)] * num_node), axis=1)
            l_t_i_1 = tf.gather(l_t_i, indices=np.array([[e]*num_node for e in range(num_node)]).reshape([-1,]), axis=1)

            tmp_e = tf.matmul(tf.reshape(tf.concat((l_t_i_0, l_t_i_1), axis=-1), [-1, units*2]), a_i)
            tmp_e = tf.nn.softmax(activation(tf.reshape(tmp_e, [-1, 1, num_node, num_node])), axis=-1)

            e_multi_head.append(tmp_e)

        alpha = tf.concat(e_multi_head, axis=1)

        # Averaging
        gc_output = activation(tf.reduce_mean(tf.matmul(alpha, tf.transpose(l_t, [0, 2, 1, 3])), axis=1))

        return alpha, gc_output

    @staticmethod
    def add_residual_ga_layer(inputs, units, num_head, activation=tf.nn.tanh):
        '''
        Call the add_ga_layer_matrix function to build the Graph Attention Layer, 
        and add the residual layer to optimize the deep neural network.
        '''
        _, gc_output = GAL.add_ga_layer_matrix(inputs, units, num_head, activation=activation)

        gc_output_residual = tf.concat([gc_output, inputs], axis=-1)

        return gc_output_residual


# Graph Convolution Layer
class GCL(object):
    '''
    This class provides static methods for adding Graph Convolution Layer.
    '''
    @staticmethod
    def add_gc_layer(inputs,
                     gcn_k,
                     laplacian_matrix,
                     output_size,
                     dtype=tf.float32,
                     use_bias=True,
                     trainable=True,
                     initializer=None,
                     regularizer=None,
                     activation=tf.nn.tanh):
        '''
        Args:
            Input(ndarray): The input features with shape [batch, num_node, num_feature].
            gcn_k(int): The highest order of Chebyshev Polynomial approximation in GCN.
            laplacian_matrix(ndarray): Laplacian matrix used in GCN, with shape [num_node, num_node].
            output_size(int): Number of output channels.
            dtype: Data type. default:tf.float32.
            use_bias(bool): It determines whether to add bias in the output. default:True.
            trainable(bool): It determines whether `weights` tensor can be trained. default:True.
            initializer: It determines whether the "weight" tensor is initialized. default:None.
            regularizer: It determines whether the "weight" tensor is regularized. default:None.
            activation(function): activation function. default:tf.nn.tanh.
        Returns:
            Returns the result of convolution of `inputs` and `laplacian_matrix`
        '''
        # [batch_size, num_node, num_feature]
        input_shape = inputs.get_shape().with_rank(3)

        num_node = tf.shape(inputs)[-2]
        num_feature = input_shape[-1].value

        # GC on inputs
        # reshape from [batch, num_node, num_feature] into [num_node, batch*num_feature]
        gc_input = tf.reshape(tf.transpose(inputs, perm=[1, 0, 2]), [num_node, -1])

        # Chebyshev polynomials
        # Reference: https://github.com/mdeff/cnn_graph
        gc_outputs = list()
        # Xt_0 = T_0 X = I X = X.
        gc_outputs.append(gc_input)
        # Xt_1 = T_1 X = L X.
        if gcn_k >= 1:
            gc_outputs.append(tf.matmul(laplacian_matrix, gc_input))
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, gcn_k+1):
            gc_outputs.append(2 * tf.matmul(laplacian_matrix, gc_outputs[-1]) - gc_outputs[-1])

        # [gcn_k+1, number_nodes, batch*num_feature]
        gc_outputs = tf.reshape(gc_outputs, [gcn_k+1, num_node, -1, num_feature])
        # [batch, number_nodes, num_feature, gcn_k+1]
        gc_outputs = tf.transpose(gc_outputs, [2, 1, 3, 0])
        # [batch*number_nodes, num_feature*gcn_k+1]
        gc_outputs = tf.reshape(gc_outputs, [-1, num_feature*(gcn_k+1)])

        output_weight = tf.get_variable("weights", shape=[num_feature*(gcn_k+1), output_size],
                                        trainable=trainable, dtype=dtype,
                                        initializer=initializer, regularizer=regularizer)
        gc_outputs = tf.matmul(gc_outputs, output_weight)

        if use_bias:
            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(0, dtype=dtype))
            gc_outputs = tf.nn.bias_add(gc_outputs, biases)

        gc_outputs = tf.reshape(gc_outputs, [-1, num_node, output_size])

        return activation(gc_outputs)

    @staticmethod
    def add_multi_gc_layers(inputs, gcn_k, gcn_l, output_size, laplacian_matrix, activation=tf.nn.tanh):
        '''
        Call add_gc_layer function to add multi Graph Convolution Layer.`gcn_l` is the number of layers added.
        '''
        with tf.variable_scope('multi_gcl', reuse=False):
            for i in range(gcn_l):
                with tf.variable_scope('gcl_%s' % i, reuse=False):
                    inputs = GCL.add_gc_layer(inputs=inputs,
                                              gcn_k=gcn_k,
                                              laplacian_matrix=laplacian_matrix,
                                              output_size=output_size,
                                              activation=activation)
        return inputs
