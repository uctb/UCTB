import numpy as np
import tensorflow as tf
import heapq

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, linalg_ops, math_ops

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr


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
        with tf.variable_scope('multi_gcl', reuse=tf.AUTO_REUSE):
            for i in range(gcn_l):
                with tf.variable_scope('gcl_%s' % i, reuse=False):
                    inputs = GCL.add_gc_layer(inputs=inputs,
                                              gcn_k=gcn_k,
                                              laplacian_matrix=laplacian_matrix,
                                              output_size=output_size,
                                              activation=activation)
        return inputs


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [
            K.in_train_phase(dropped_inputs, ones, training=training)
            for _ in range(count)
        ]
    return K.in_train_phase(dropped_inputs, ones, training=training)


class GCLSTMCell(tf.keras.layers.LSTMCell):
    """
    GCLSTMCell is one of our implemented ST-RNN models in handling the spatial and temporal features.
    We performed GCN on both LSTM inputs and hidden-states. The code is inherited from tf.keras.layers.LSTMCell,
    thus it can be used almost the same as LSTMCell except that you need to provide the GCN parameters
    in the __init__ function.

    Args:
        units(int): number of units of LSTM
        num_nodes(int): number of nodes in the graph
        laplacian_matrix(ndarray): laplacian matrix used in GCN, with shape [num_node, num_node]
        gcn_k(int): highest order of Chebyshev Polynomial approximation in GCN
        gcn_l(int): number of GCN layers
        kwargs: other parameters supported by LSTMCell, such as activation, kernel_initializer ... and so on.
    """

    def __init__(self, units, num_nodes, laplacian_matrix, gcn_k=1, gcn_l=1, **kwargs):

        super().__init__(units, **kwargs)

        self._units = units
        self._num_node = num_nodes
        self._gcn_k = gcn_k
        self._gcn_l = gcn_l
        self._laplacian_matrix = laplacian_matrix

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super(GCLSTMCell, self).build(input_shape)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim * (self._gcn_k + 1), self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units * (self._gcn_k + 1), self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

    def kth_cheby_ploy(self, k, tk1=None, tk2=None):
        if k == 0:
            return linalg_ops.eye(self._num_node, dtype=dtypes.float32)
        elif k == 1:
            return self._laplacian_matrix
        elif k > 1:
            return math_ops.matmul(2 * self._laplacian_matrix, tk1) - tk2

    def call(self, inputs, states, training=None):

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        input_dim = inputs.get_shape()[-1].value

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0. < self.dropout < 1.:
            inputs *= dp_mask[0]
        if 0. < self.recurrent_dropout < 1.:
            h_tm1 *= rec_dp_mask[0]

        # inputs has shape: [batch * num_nodes, input_dim]
        # h_tm1 has shape: [batch * num_nodes, units]
        inputs_before_gcn = tf.reshape(tf.transpose(tf.reshape(inputs, [-1, self._num_node, input_dim]),
                                                    [1, 0, 2]), [self._num_node, -1])
        h_tm1_before_gcn = tf.reshape(tf.transpose(tf.reshape(h_tm1, [-1, self._num_node, self._units]),
                                                   [1, 0, 2]), [self._num_node, -1])

        t = []
        inputs_after_gcn = list()
        h_tm1_after_gcn = list()
        for i in range(0, self._gcn_k + 1):
            t.append(self.kth_cheby_ploy(k=i, tk1=None if i < 1 else t[i - 1], tk2=None if i < 2 else t[i - 2]))
            inputs_after_gcn.append(tf.matmul(t[-1], inputs_before_gcn))
            h_tm1_after_gcn.append(tf.matmul(t[-1], h_tm1_before_gcn))

        inputs_after_gcn = tf.reshape(inputs_after_gcn, [self._gcn_k + 1, self._num_node, -1, input_dim])
        h_tm1_after_gcn = tf.reshape(h_tm1_after_gcn, [self._gcn_k + 1, self._num_node, -1, self._units])

        inputs_after_gcn = tf.reshape(tf.transpose(inputs_after_gcn, [2, 1, 0, 3]), [-1, (self._gcn_k + 1) * input_dim])
        h_tm1_after_gcn = tf.reshape(tf.transpose(h_tm1_after_gcn, [2, 1, 0, 3]), [-1, (self._gcn_k + 1) * self.units])

        z = K.dot(inputs_after_gcn, self.kernel)
        z += K.dot(h_tm1_after_gcn, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z0 = z[:, :self.units]  # i=self.recurrent_activation(z0)
        z1 = z[:, self.units:2 * self.units]  # f=self.recurrent_activation(z1)
        z2 = z[:, 2 * self.units:3 * self.units]  # c=f*c_tm1 + i*self.activation(z2)
        z3 = z[:, 3 * self.units:]  # o=self.recurrent_activation(z3)

        z = (z0, z1, z2, z3)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]


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

