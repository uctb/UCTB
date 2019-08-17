import numpy as np
import tensorflow as tf

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr


class GraphBuilder(object):
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
        adjacent_matrix = np.zeros([traffic_data.shape[1], traffic_data.shape[1]])
        for i in range(traffic_data.shape[1]):
            for j in range(traffic_data.shape[1]):
                r, p_value = pearsonr(traffic_data[:, i], traffic_data[:, j])
                adjacent_matrix[i, j] = 0 if np.isnan(r) else r
        adjacent_matrix = (adjacent_matrix >= threshold).astype(np.float32)
        return adjacent_matrix

    @staticmethod
    def distance_adjacent(lat_lng_list, threshold):
        adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
        for i in range(len(lat_lng_list)):
            for j in range(len(lat_lng_list)):
                adjacent_matrix[i][j] = GraphBuilder.haversine(lat_lng_list[i][0], lat_lng_list[i][1],
                                                               lat_lng_list[j][0], lat_lng_list[j][1])
        adjacent_matrix = (adjacent_matrix <= threshold).astype(np.float32)
        return adjacent_matrix

    @staticmethod
    def interaction_adjacent(interaction_matrix, threshold):
        return (interaction_matrix >= threshold).astype(np.float32)

    @staticmethod
    def adjacent_to_laplacian(adjacent_matrix):
        adjacent_matrix -= np.diag(np.diag(adjacent_matrix))
        diagonal_matrix = np.diag(np.sum(adjacent_matrix, axis=0) ** -0.5)
        diagonal_matrix[np.isinf(diagonal_matrix)] = 0
        laplacian_matrix = np.eye(len(adjacent_matrix)) - np.dot(np.dot(diagonal_matrix, adjacent_matrix),
                                                                 diagonal_matrix)
        laplacian_matrix = 2 * laplacian_matrix / np.max(laplacian_matrix) - np.eye(len(adjacent_matrix))
        return laplacian_matrix


# Graph Attention Layer
class GAL(object):

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

        _, gc_output = GAL.add_ga_layer_matrix(inputs, units, num_head, activation=activation)

        gc_output_residual = tf.concat([gc_output, inputs], axis=-1)

        return gc_output_residual


# Graph Convolution Layer
class GCL(object):

    @staticmethod
    def KthChebyPloy(k, num_nodes, laplacian_matrix, T_k_1=None, T_k_2=None):
        if k == 0:
            return tf.eye(num_nodes, dtype=tf.float32)
        elif k == 1:
            return laplacian_matrix
        elif k > 1:
            return tf.matmul(2 * laplacian_matrix, T_k_1) - T_k_2

    @staticmethod
    def add_gc_layer(inputs, K, laplacian_matrix, activation=tf.nn.tanh):

        # [-1, num_node, num_feature]
        input_shape = inputs.get_shape().with_rank(3)

        num_node = tf.shape(inputs)[-2]
        num_feature = input_shape[-1].value

        # GC on inputs
        # reshape from [batch, num_node, num_feature] into [num_node, batch*num_feature]
        gc_input = tf.reshape(tf.transpose(inputs, perm=[1, 0, 2]), [num_node, -1])

        theta = tf.Variable(tf.random_normal([K + 1, ], dtype=tf.float32))

        chebyPly_inputs = []
        T = []
        for i in range(0, K + 1):
            T.append(GCL.KthChebyPloy(i, num_node, laplacian_matrix,
                                      None if i < 1 else T[i - 1], None if i < 2 else T[i - 2]))
            chebyPly_inputs.append(theta[i] * tf.matmul(T[-1], gc_input))

        gc_output = tf.tanh(tf.reduce_sum(chebyPly_inputs, axis=0))

        gc_output = tf.transpose(tf.reshape(gc_output, [num_node, -1, num_feature]), perm=[1, 0, 2])

        return activation(gc_output)

    @staticmethod
    def add_multi_gc_layers(inputs, K, L, laplacian_matrix, activation=tf.nn.tanh):
        with tf.variable_scope('multi_gcl', reuse=False):
            for i in range(L):
                inputs = GCL.add_gc_layer(inputs, K, laplacian_matrix, activation)
        return inputs


if __name__ == '__main__':

    graph = tf.Graph()

    with graph.as_default():
        a = tf.placeholder(tf.float32, [300, 7, 6])
        print(GAL.add_ga_layer(a, units=16, num_head=8))