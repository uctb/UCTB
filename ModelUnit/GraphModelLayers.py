import numpy as np
import tensorflow as tf

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr


class GraphBuilder(object):
    @staticmethod
    def correlation_graph(traffic_data, threshold=0, keep_weight=True):
        A = np.zeros([traffic_data.shape[1], traffic_data.shape[1]])
        D = np.eye(traffic_data.shape[1])
        for i in range(traffic_data.shape[1]):
            for j in range(traffic_data.shape[1]):
                if i == j:
                    continue
                r, p_value = pearsonr(traffic_data[:, i], traffic_data[:, j])
                # set 0 for nan and negative value
                if np.isnan(r) or r <= threshold:
                    r = 0
                if not keep_weight:
                    r = 1
                A[i, j] = r
            D[i, i] = 1 if np.sum(A[i, :]) == 0 else np.sum(A[i, :])

        D_Normal = np.linalg.inv(D) ** 0.5

        LM = np.eye(traffic_data.shape[1]) - np.dot(np.dot(D_Normal, A), D_Normal)

        LM = 2 * LM / np.max(LM) - np.eye(len(A))

        return LM

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
        r = 6371  # 地球平均半径，单位为公里

        return c * r * 1000

    @staticmethod
    def distance_graph(lat_lng_list, threshold=1000):
        A = np.zeros([len(lat_lng_list), len(lat_lng_list)])
        D = np.eye(len(lat_lng_list))
        for i in range(len(lat_lng_list)):
            for j in range(len(lat_lng_list)):
                if i == j:
                    continue
                distance = GraphBuilder.haversine(lat_lng_list[i][0], lat_lng_list[i][1],
                                                  lat_lng_list[j][0], lat_lng_list[j][1])
                if distance < threshold:
                    A[i][j] = 1
            D[i, i] = 1 if np.sum(A[i, :]) == 0 else np.sum(A[i, :])

        D_Normal = np.linalg.inv(D) ** 0.5

        LM = np.eye(len(lat_lng_list)) - np.dot(np.dot(D_Normal, A), D_Normal)

        LM = 2 * LM / np.max(LM) - np.eye(len(A))

        return LM

    @staticmethod
    def interaction_graph(interaction_matrix, threshold=500):
        A = np.zeros([len(interaction_matrix), len(interaction_matrix)])
        D = np.eye(len(interaction_matrix))
        for i in range(len(interaction_matrix)):
            for j in range(len(interaction_matrix)):
                if i == j:
                    continue
                interaction = interaction_matrix[i][j]
                if interaction > threshold:
                    A[i][j] = 1
            D[i, i] = 1 if np.sum(A[i, :]) == 0 else np.sum(A[i, :])

        D_Normal = np.linalg.inv(D) ** 0.5

        LM = np.eye(len(interaction_matrix)) - np.dot(np.dot(D_Normal, A), D_Normal)

        LM = 2 * LM / np.max(LM) - np.eye(len(A))

        return LM

    @staticmethod
    def adjacent_to_lm(matrix):
        A = np.zeros([len(matrix), len(matrix)])
        D = np.eye(len(matrix))
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i == j:
                    continue
                A[i][j] = matrix[i][j]
            D[i, i] = 1 if np.sum(A[i, :]) == 0 else np.sum(A[i, :])
        D_Normal = np.linalg.inv(D) ** 0.5
        LM = np.eye(len(matrix)) - np.dot(np.dot(D_Normal, A), D_Normal)
        LM = 2 * LM / np.max(LM) - np.eye(len(A))
        return LM


# Graph Attention Layer
class GAL(object):
    @staticmethod
    def add_ga_layer(graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu, with_self_loop=True):
        
        inputs = graph.get_tensor_by_name(inputs_name)

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
        for i in range(num_node):
            for j in range(num_node):
                if i == j and not with_self_loop:
                    continue
                multi_head_result = []
                for k in range(num_head):
                    multi_head_result.append(tf.matmul(tf.concat([l_t[:, i, k, :], l_t[:, j, k, :]], axis=-1),
                                                       a[:, k:k+1]))
                e.append(tf.reshape(tf.concat(multi_head_result, axis=-1), [-1, num_head, 1]))

        e = activation(tf.reshape(tf.concat(e, axis=-1), [-1, num_head, num_node,
                                                          num_node if with_self_loop else num_node-1]))

        alpha = tf.nn.softmax(e, axis=-1)

        if not with_self_loop:
            return alpha.name

        # Averaging
        gc_output = tf.reduce_mean(tf.matmul(alpha, tf.transpose(l_t, [0, 2, 1, 3])), axis=1)

        return alpha.name, gc_output.name

    @staticmethod
    def add_ga_layer_matrix(graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu, with_self_loop=True):

        inputs = graph.get_tensor_by_name(inputs_name)

        inputs_shape = inputs.get_shape().with_rank(3)

        num_node = inputs_shape[-2].value
        num_feature = inputs_shape[-1].value

        W = tf.Variable(tf.random_normal([num_feature, units * num_head]))

        # linear transform
        l_t = tf.matmul(tf.reshape(inputs, [-1, num_feature]), W)
        l_t = tf.reshape(l_t, [-1, num_node, num_head, units])

        a = tf.Variable(tf.random_normal([units * 2, num_head]))

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
        gc_output = tf.reduce_mean(tf.matmul(alpha, tf.transpose(l_t, [0, 2, 1, 3])), axis=1)

        return alpha.name, gc_output.name

    @staticmethod
    def add_residual_ga_layer(graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu):

        _, gc_output_name = GAL.add_ga_layer_matrix(graph, inputs_name, units, num_head,
                                                    activation=activation, with_self_loop=True)

        inputs = graph.get_tensor_by_name(inputs_name)

        gc_output_residual = tf.concat([graph.get_tensor_by_name(gc_output_name), inputs], axis=-1)

        return gc_output_residual.name


# Graph Convolution Layer
class GCL(object):

    @staticmethod
    def KthChebyPloy(k, num_nodes, laplacian_matrix, T_k_1=None, T_k_2=None):
        if k == 0:
            return tf.eye(num_nodes)
        elif k == 1:
            return laplacian_matrix
        elif k > 1:
            return tf.matmul(2 * laplacian_matrix, T_k_1) - T_k_2

    @staticmethod
    def add_gc_layer(graph, inputs_name, K, laplacian_matrix):

        inputs = graph.get_tensor_by_name(inputs_name)

        # [-1, num_node, num_feature]
        input_shape = inputs.get_shape().with_rank(3)

        num_node = tf.shape(inputs)[-2]
        num_feature = input_shape[-1].value

        # GC on inputs
        # reshape from [batch, num_node, num_feature] into [num_node, batch*num_feature]
        gc_input = tf.reshape(tf.transpose(inputs, perm=[1, 0, 2]), [num_node, -1])

        theta = tf.Variable(tf.random_normal([K + 1, ]))

        chebyPly_inputs = []
        T = []
        for i in range(0, K + 1):
            T.append(GCL.KthChebyPloy(i, num_node, laplacian_matrix,
                                      None if i < 1 else T[i - 1], None if i < 2 else T[i - 2]))
            chebyPly_inputs.append(theta[i] * tf.matmul(T[-1], gc_input))

        gc_output = tf.tanh(tf.reduce_sum(chebyPly_inputs, axis=0))

        gc_output = tf.transpose(tf.reshape(gc_output, [num_node, -1, num_feature]), perm=[1, 0, 2])

        return gc_output.name


if __name__ == '__main__':

    graph = tf.Graph()

    with graph.as_default():
        a = tf.placeholder(tf.float32, [300, 7, 6])
        print(GAL.add_ga_layer(graph, a.name, units=16, num_head=8))