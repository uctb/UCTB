import os
import numpy as np
import tensorflow as tf
import shutil

from Model.GCN_LSTM_CELL import GCN_LSTMCell
from Model.GraphModelLayers import GAL

from Model.BaseModel import BaseModel


class MGCNRegression(BaseModel):
    def __init__(self,
                 num_node,
                 GCN_K,
                 GCN_layers,
                 num_graph,
                 external_dim,
                 T,
                 num_hidden_units,
                 num_filter_conv1x1,
                 lr,
                 code_version,
                 model_dir,
                 GPU_DEVICE='0'):

        super(MGCNRegression, self).__init__(code_version=code_version,
                                             model_dir=model_dir,
                                             GPU_DEVICE=GPU_DEVICE)

        self._num_node = num_node
        self._gcn_k = GCN_K
        self._gcn_layer = GCN_layers
        self._num_graph = num_graph
        self._external_dim = external_dim

        self._T = T
        self._num_hidden_unit = num_hidden_units
        self._num_filter_conv1x1 = num_filter_conv1x1
        self._lr = lr

    def build(self):
        with self._graph.as_default():

            # Input
            input = tf.placeholder(tf.float32, [None, self._T, None, 1], name='input')
            target = tf.placeholder(tf.float32, [None, None], name='target')
            laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')

            batch_size = tf.shape(input)[0]

            # recode input
            self._input['input'] = input.name
            self._input['target'] = target.name
            self._input['laplace_matrix'] = laplace_matrix.name

            outputs_last_list = []

            for graph_index in range(self._num_graph):
                with tf.variable_scope('gc_lstm_%s' % graph_index, reuse=False):
                    outputs_all = []
                    # final_state_all = []

                    if type(self._gcn_k) is list:
                        gc_lstm_1 = GCN_LSTMCell(self._gcn_k[graph_index], self._gcn_layer[graph_index], self._num_node,
                                             self._num_hidden_unit, state_is_tuple=True,
                                             initializer=tf.contrib.layers.xavier_initializer())
                    else:
                        gc_lstm_1 = GCN_LSTMCell(self._gcn_k, self._gcn_layer, self._num_node,
                                             self._num_hidden_unit, state_is_tuple=True,
                                             initializer=tf.contrib.layers.xavier_initializer())

                    gc_lstm_1.laplacian_matrix = tf.transpose(laplace_matrix[graph_index])

                    state_1 = gc_lstm_1.zero_state(batch_size, dtype=tf.float32)

                    for i in range(0, self._T):

                        output_1, state_1 = gc_lstm_1(input[:, i, :, :], state_1)

                        outputs_all.append(output_1)
                        # final_state_all.append(state)

                outputs_last_list.append(tf.reshape(outputs_all[-1], [-1, 1, self._num_hidden_unit]))

            if self._num_graph > 1:
                # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                _, gal_output_name = GAL.add_ga_layer(graph=self._graph,
                                                      inputs_name=tf.concat(outputs_last_list, axis=-2).name,
                                                      units=64, num_head=2, with_self_loop=True)
                gal_output = self._graph.get_tensor_by_name(gal_output_name)
                pre_input = tf.reshape(tf.reduce_mean(gal_output, axis=-2),
                                       [-1, self._num_node, 1, self._num_hidden_unit])
            else:
                pre_input = tf.reshape(outputs_all[-1], [-1, self._num_node, 1, self._num_hidden_unit])

            pre_input = tf.layers.batch_normalization(pre_input, axis=-1)

            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, tf.shape(pre_input)[1], tf.shape(pre_input)[2], 1])
                pre_input = tf.concat([pre_input, external_dense], axis=-1)

            conv1x1_output0 = tf.layers.conv2d(pre_input,
                                               filters=self._num_filter_conv1x1,
                                               kernel_size=[1, 1],
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            pre_output = tf.layers.conv2d(conv1x1_output0,
                                          filters=1,
                                          kernel_size=[1, 1],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            prediction = tf.reshape(pre_output, [batch_size, self._num_node], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_operation = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_operation.name

            self._saver = tf.train.Saver(max_to_keep=None)
            self._variable_init = tf.global_variables_initializer()

        self._session.run(self._variable_init)
        self._build = False

    def fit(self, X, y, l_m, external_feature=None):
        if hasattr(X, 'shape') is False or hasattr(y, 'shape') is False or hasattr(l_m, 'shape') is False:
            raise ValueError('Please feed numpy array')

        if X.shape[0] != y.shape[0]:
            raise ValueError('Expected X and y have the same batch_size, but given', X.shape, y.shape)

        feed_dict = {
            self._graph.get_tensor_by_name(self._input['input']): X,
            self._graph.get_tensor_by_name(self._input['target']): y,
            self._graph.get_tensor_by_name(self._input['laplace_matrix']): l_m
        }

        if self._external_dim is not None and self._external_dim > 0:
            feed_dict[self._graph.get_tensor_by_name(self._input['external_input'])] = external_feature

        l, _ = self._session.run([self._graph.get_tensor_by_name(self._output['loss']),
                                  self._graph.get_operation_by_name(self._op['train_op'])],
                                 feed_dict=feed_dict)
        return l

    def predict(self, X, l_m, external_feature=None):
        feed_dict = {
            self._graph.get_tensor_by_name(self._input['input']): X,
            self._graph.get_tensor_by_name(self._input['laplace_matrix']): l_m
        }

        if self._external_dim is not None and self._external_dim > 0:
            feed_dict[self._graph.get_tensor_by_name(self._input['external_input'])] = external_feature

        p = self._session.run(self._graph.get_tensor_by_name(self._output['prediction']),
                              feed_dict=feed_dict)
        return p

    def evaluate(self, X, y, l_m, metric, cache_volume=64, external_feature=None, threshold=0, de_normalizer=None):
        p = []
        for i in range(0, len(X), cache_volume):
            p.append(self.predict(X[i:i + cache_volume],
                                  l_m, external_feature[i:i + cache_volume]))
        p = np.concatenate(p, axis=0)
        if de_normalizer is not None:
            p = de_normalizer(p)
            y = de_normalizer(y)
        return [e(p, y, threshold=threshold) for e in metric]