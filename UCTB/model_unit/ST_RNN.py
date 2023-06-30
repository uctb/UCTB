import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, linalg_ops, math_ops


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
        num_node(int): number of nodes in the graph
        laplacian_matrix(ndarray): laplacian matrix used in GCN, with shape [num_node, num_node]
        gcn_k(int): highest order of Chebyshev Polynomial approximation in GCN
        gcn_l(int): number of GCN layers
        kwargs: other parameters supported by LSTMCell, such as activation, kernel_initializer ... and so on.
    """

    def __init__(self, units, num_node, laplacian_matrix, gcn_k=1, gcn_l=1, **kwargs):

        super().__init__(units, **kwargs)

        self._units = units
        self._num_node = num_node
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

        # inputs has shape: [batch * num_node, input_dim]
        # h_tm1 has shape: [batch * num_node, units]
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

        z0 = z[:, :self.units]
        z1 = z[:, self.units:2 * self.units]
        z2 = z[:, 2 * self.units:3 * self.units]
        z3 = z[:, 3 * self.units:]

        z = (z0, z1, z2, z3)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]
