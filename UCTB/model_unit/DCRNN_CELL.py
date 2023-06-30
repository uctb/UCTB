from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, input_dim, num_graphs, supports, max_diffusion_step, num_node, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, use_gc_for_ru=True, name=None):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_node:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_node = num_node
        self._input_dim = input_dim
        self._num_graphs = num_graphs
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._supports = supports
        self._num_diff_matrix = supports.get_shape()[0].value
        self._name = name

    @property
    def state_size(self):
        return self._num_node * self._num_units

    @property
    def output_size(self):
        output_size = self._num_node * self._num_units
        if self._num_proj is not None:
            output_size = self._num_node * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_node * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                if self._use_gc_for_ru:
                    fn = self._gconv
                else:
                    fn = self._fc
                value = tf.nn.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
                value = tf.reshape(value, (-1, self._num_node, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_node * self._num_units))
                u = tf.reshape(u, (-1, self._num_node * self._num_units))
            with tf.variable_scope("candidate"):
                c = self._gconv(inputs, r * state, self._num_units)
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(-1, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_node, -1))
        state = tf.reshape(state, (batch_size * self._num_node, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_node, input_dim/state_dim)
        last_dim = inputs.get_shape()[-1].value
        inputs = tf.reshape(inputs, (-1, self._num_node, int(last_dim / self._num_node)))
        state = tf.reshape(state, (-1, self._num_node, self._num_units))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_node, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_node, -1])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope.name + (self.name or ''), reuse=False):
            if self._max_diffusion_step == 0:
                pass
            else:
                for index in range(self._num_diff_matrix):
                    x1 = tf.matmul(self._supports[index], x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.matmul(self._supports[index], x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = self._num_diff_matrix * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_node, input_size, -1])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_node, input_size, order)
            x = tf.reshape(x, shape=[-1, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_node, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [-1, self._num_node * output_size])
