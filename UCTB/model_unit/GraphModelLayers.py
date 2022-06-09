import numpy as np
import tensorflow as tf
import heapq

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr

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
    def add_multi_gc_layers(inputs, graph_id, gcn_k, gcn_l, output_size, laplacian_matrix, activation=tf.nn.tanh):
        '''
        Call add_gc_layer function to add multi Graph Convolution Layer.`gcn_l` is the number of layers added.
        '''
        with tf.variable_scope('multi_gcl_%s' % graph_id, reuse=False):
            for i in range(gcn_l):
                with tf.variable_scope('gcl_%s' % i, reuse=False):
                    inputs = GCL.add_gc_layer(inputs=inputs,
                                              gcn_k=gcn_k,
                                              laplacian_matrix=laplacian_matrix,
                                              output_size=output_size,
                                              activation=activation)
        return inputs
