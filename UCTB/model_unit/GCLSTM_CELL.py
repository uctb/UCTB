from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, LSTMStateTuple, _concat

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class GCLSTMCell(LayerRNNCell):
    def __init__(self, K, num_layers, num_nodes, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None, name=None):

        super(GCLSTMCell, self).__init__(_reuse=reuse, name=name)

        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        if K < 0:
            raise ValueError("K should greater than 0")

        # Inputs must be 3-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=3)
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._num_layers = num_layers
        # self._num_target = num_target
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._laplacian_matrix = None
        self._K = K

        if num_proj:
            self._state_size = (
                LSTMStateTuple([num_nodes, num_units], [num_nodes, num_proj])
                if state_is_tuple else [num_nodes, num_units + num_proj])
            self._output_size = [num_nodes, num_proj]
        else:
            self._state_size = (
                LSTMStateTuple([num_nodes, num_units], [num_nodes, num_units])
                if state_is_tuple else [num_nodes, 2 * num_units])
            self._output_size = [num_nodes, num_units]

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def laplacian_matrix(self):
        return self._laplacian_matrix

    @laplacian_matrix.setter
    def laplacian_matrix(self, value):
        self._laplacian_matrix = value

    def build(self, inputs_shape):
        if inputs_shape[2].value is None:
            raise ValueError("Expected inputs.shape[-2] to be known, saw shape: %s"
                             % inputs_shape)

        if self.laplacian_matrix is None:
            raise ValueError("Please feed laplacian matrix")

        input_depth = inputs_shape[2].value

        h_depth = self._num_units if self._num_proj is None else self._num_proj

        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)

        # f, i, j, o
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=self._initializer,
            partitioner=maybe_partitioner)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._theta = self.add_variable(
            'ChebyshevParameter',
            shape=[self._num_layers, self._K + 1],
            initializer=self._initializer)

        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True

    def KthChebyPloy(self, k, T_k_1=None, T_k_2=None):
        if k == 0:
            return linalg_ops.eye(self._num_nodes)
        elif k == 1:
            return self._laplacian_matrix
        elif k > 1:
            return math_ops.matmul(2 * self._laplacian_matrix, T_k_1) - T_k_2

    def zero_state(self, batch_size, dtype):
        state_size = self.state_size
        is_eager = context.executing_eagerly()
        if is_eager and hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype,
             last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and
                    last_dtype == dtype and
                    last_state_size == state_size):
                return last_output
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            c = _concat(batch_size, state_size[0])
            c_size = array_ops.zeros(c, dtype=dtype)
            if not context.executing_eagerly():
                c_static = _concat(batch_size, state_size[0], static=True)
                c_size.set_shape(c_static)
            h = _concat(batch_size, state_size[1])
            h_size = array_ops.zeros(h, dtype=dtype)
            if not context.executing_eagerly():
                h_static = _concat(batch_size, state_size[1], static=True)
                h_size.set_shape(h_static)
            output = nest._sequence_like(state_size, [c_size, h_size])
        if is_eager:
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    def call(self, inputs, state):
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0, 0], [-1, -1, self._num_units])
            m_prev = array_ops.slice(state, [0, 0, self._num_units], [-1, -1, num_proj])

        input_size = inputs.get_shape().with_rank(3)[2]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        ##############################################################################################################
        # GC on inputs
        # reshape from [batch, num_node, num_feature] into [num_node, batch*num_feature]
        x0 = array_ops.reshape(array_ops.transpose(inputs, perm=[1, 0, 2]), [self._num_nodes, -1])
        h0 = array_ops.reshape(array_ops.transpose(m_prev, perm=[1, 0, 2]), [self._num_nodes, -1])
        for layer_index in range(self._num_layers):
            chebyPly_X = []
            chebyPly_H = []
            T = []
            for i in range(0, self._K + 1):
                T.append(self.KthChebyPloy(i, None if i < 1 else T[i - 1], None if i < 2 else T[i - 2]))
                chebyPly_X.append(self._theta[layer_index][i] * math_ops.matmul(T[-1], x0))
                chebyPly_H.append(self._theta[layer_index][i] * math_ops.matmul(T[-1], h0))
            if layer_index > self._num_layers - 1:
                x0 = math_ops.tanh(math_ops.reduce_sum(chebyPly_X, axis=0))
                h0 = math_ops.tanh(math_ops.reduce_sum(chebyPly_H, axis=0))
            else:
                x0 = math_ops.reduce_sum(chebyPly_X, axis=0)
                h0 = math_ops.reduce_sum(chebyPly_H, axis=0)

        x0 = array_ops.transpose(array_ops.reshape(x0, [self._num_nodes, -1, input_size]), perm=[1, 0, 2])
        h0 = array_ops.transpose(array_ops.reshape(h0, [self._num_nodes, -1, self._num_units]), perm=[1, 0, 2])

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = math_ops.matmul(
            array_ops.reshape(array_ops.concat([x0, h0], 2), [-1, input_size + self._num_units]), self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        lstm_matrix = array_ops.reshape(lstm_matrix, [-1, self._num_nodes, self._kernel.shape[-1]])

        i, j, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=-1)

        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))

        return m, new_state