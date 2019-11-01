import keras
import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import GAL, GCL
from ..model_unit import DCGRUCell
from ..model_unit import GCLSTMCell


class AMultiGCLSTM(BaseModel):
    def __init__(self,
                 num_node,
                 external_dim,
                 closeness_len,
                 period_len,
                 trend_len,

                 # gcn parameters
                 num_graph=1,
                 gcn_k=1,
                 gcn_layers=1,
                 gclstm_layers=1,

                 # dense units
                 num_hidden_units=64,
                 # LSTM units
                 num_filter_conv1x1=32,

                 # temporal attention parameters
                 tpe_dim=None,
                 temporal_gal_units=32,
                 temporal_gal_num_heads=2,
                 temporal_gal_layers=4,

                 # merge parameters
                 graph_merge_gal_units=32,
                 graph_merge_gal_num_heads=2,
                 temporal_merge_gal_units=64,
                 temporal_merge_gal_num_heads=2,

                 # network structure parameters
                 st_method='GCLSTM',       # gclstm
                 temporal_merge='gal',     # gal
                 graph_merge='gal',        # concat

                 output_activation=tf.nn.sigmoid,

                 # Transfer learning
                 build_sd_regularization=False,
                 build_transfer=False,
                 transfer_ratio=0,

                 lr=1e-4,
                 code_version='AMulti-QuickStart',
                 model_dir='model_dir',
                 gpu_device='0', **kwargs):

        super(AMultiGCLSTM, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)

        self._num_node = num_node
        self._gcn_k = gcn_k
        self._gcn_layer = gcn_layers
        self._graph_merge_gal_units = graph_merge_gal_units
        self._graph_merge_gal_num_heads = graph_merge_gal_num_heads
        self._temporal_merge_gal_units = temporal_merge_gal_units
        self._temporal_merge_gal_num_heads = temporal_merge_gal_num_heads
        self._gclstm_layers = gclstm_layers
        self._temporal_gal_units = temporal_gal_units
        self._temporal_gal_num_heads = temporal_gal_num_heads
        self._temporal_gal_layers = temporal_gal_layers
        self._num_graph = num_graph
        self._external_dim = external_dim
        self._output_activation = output_activation

        self._st_method = st_method
        self._temporal_merge = temporal_merge
        self._graph_merge = graph_merge

        self._build_transfer = build_transfer
        self._build_sd_regularization = build_sd_regularization
        self._transfer_ratio = transfer_ratio
        if self._build_transfer:
            assert 0 <= self._transfer_ratio <= 1

        self._tpe_dim = tpe_dim
        if st_method == 'gal_gcn':
            assert self._tpe_dim

        self._closeness_len = int(closeness_len)
        self._period_len = int(period_len)
        self._trend_len = int(trend_len)
        self._num_hidden_unit = num_hidden_units
        self._num_filter_conv1x1 = num_filter_conv1x1
        self._lr = lr
    
    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():

            temporal_features = []

            if self._closeness_len is not None and self._closeness_len > 0:
                if self._st_method == 'gclstm' or 'DCRNNN':
                    closeness_feature = tf.placeholder(tf.float32, [None, None, self._closeness_len, 1],
                                                       name='closeness_feature')
                elif self._st_method == 'gal_gcn':
                    closeness_feature = tf.placeholder(tf.float32, [None, None, self._closeness_len, 1 + self._tpe_dim],
                                                       name='closeness_feature')
                self._input['closeness_feature'] = closeness_feature.name
                temporal_features.append([self._closeness_len, closeness_feature, 'closeness_feature'])

            if self._period_len is not None and self._period_len > 0:
                if self._st_method == 'gclstm' or 'DCRNNN':
                    period_feature = tf.placeholder(tf.float32, [None, None, self._period_len, 1],
                                                    name='period_feature')
                elif self._st_method == 'gal_gcn':
                    period_feature = tf.placeholder(tf.float32, [None, None, self._period_len, 1 + self._tpe_dim],
                                                    name='period_feature')
                self._input['period_feature'] = period_feature.name
                temporal_features.append([self._period_len, period_feature, 'period_feature'])

            if self._trend_len is not None and self._trend_len > 0:
                if self._st_method == 'gclstm' or 'DCRNNN':
                    trend_feature = tf.placeholder(tf.float32, [None, None, self._trend_len, 1],
                                                   name='trend_feature')
                elif self._st_method == 'gal_gcn':
                    trend_feature = tf.placeholder(tf.float32, [None, None, self._trend_len, 1 + self._tpe_dim],
                                                   name='trend_feature')
                self._input['trend_feature'] = trend_feature.name
                temporal_features.append([self._trend_len, trend_feature, 'trend_feature'])

            if len(temporal_features) > 0:
                target = tf.placeholder(tf.float32, [None, None, 1], name='target')
                laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')
                self._input['target'] = target.name
                self._input['laplace_matrix'] = laplace_matrix.name
            else:
                raise ValueError('closeness_len, period_len, trend_len cannot all be zero')

            graph_outputs_list = []

            for graph_index in range(self._num_graph):

                if self._st_method in ['GCLSTM', 'DCRNN', 'GRU', 'LSTM']:

                    outputs_temporal = []

                    for time_step, target_tensor, given_name in temporal_features:

                        if self._st_method == 'GCLSTM':

                            multi_layer_cell = tf.keras.layers.StackedRNNCells(
                                [GCLSTMCell(units=self._num_hidden_unit, num_nodes=self._num_node,
                                            laplacian_matrix=laplace_matrix[graph_index],
                                            gcn_k=self._gcn_k, gcn_l=self._gcn_layer)
                                 for _ in range(self._gclstm_layers)])

                            outputs = tf.keras.layers.RNN(multi_layer_cell)(tf.reshape(target_tensor, [-1, time_step, 1]))

                            st_outputs = tf.reshape(outputs, [-1, 1, self._num_hidden_unit])

                        elif self._st_method == 'DCRNN':

                            cell = DCGRUCell(self._num_hidden_unit, 1, self._num_graph,
                                             # laplace_matrix will be diffusion_matrix when self._st_method == 'DCRNN'
                                             laplace_matrix,
                                             max_diffusion_step=self._gcn_k,
                                             num_nodes=self._num_node, name=str(graph_index) + given_name)

                            encoding_cells = [cell] * self._gclstm_layers
                            encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)

                            inputs_unstack = tf.unstack(tf.reshape(target_tensor, [-1, self._num_node, time_step]),
                                                        axis=-1)

                            outputs, _ = \
                                tf.contrib.rnn.static_rnn(encoding_cells, inputs_unstack, dtype=tf.float32)

                            st_outputs = tf.reshape(outputs[-1], [-1, 1, self._num_hidden_unit])

                        elif self._st_method == 'GRU':

                            cell = tf.keras.layers.GRUCell(units=self._num_hidden_unit)
                            multi_layer_gru = tf.keras.layers.StackedRNNCells([cell] * self._gclstm_layers)
                            outputs = tf.keras.layers.RNN(multi_layer_gru)(
                                tf.reshape(target_tensor, [-1, time_step, 1]))
                            st_outputs = tf.reshape(outputs, [-1, 1, self._num_hidden_unit])

                        elif self._st_method == 'LSTM':

                            cell = tf.keras.layers.LSTMCell(units=self._num_hidden_unit)
                            multi_layer_gru = tf.keras.layers.StackedRNNCells([cell] * self._gclstm_layers)
                            outputs = tf.keras.layers.RNN(multi_layer_gru)(
                                tf.reshape(target_tensor, [-1, time_step, 1]))
                            st_outputs = tf.reshape(outputs, [-1, 1, self._num_hidden_unit])

                        outputs_temporal.append(st_outputs)

                    if self._temporal_merge == 'concat':
                        
                        graph_outputs_list.append(tf.concat(outputs_temporal, axis=-1))

                    elif self._temporal_merge == 'gal':

                        _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(outputs_temporal, axis=-2),
                                                                units=self._temporal_merge_gal_units,
                                                                num_head=self._temporal_merge_gal_num_heads)

                        graph_outputs_list.append(tf.reduce_mean(gal_output, axis=-2, keepdims=True))

                # elif self._st_method == 'gal_gcn':
                #
                #     attention_input = tf.reshape(tf.concat([e[1] for e in temporal_features], axis=-2),
                #                                  [-1, sum([e[0] for e in temporal_features]), 1 + self._tpe_dim])
                #     attention_output_list = []
                #     for loop_index in range(self._temporal_gal_layers):
                #         with tf.variable_scope('res_temporal_gal_%s' % loop_index, reuse=False):
                #             attention_input = GAL.add_residual_ga_layer(attention_input,
                #                                                         num_head=self._temporal_gal_num_heads,
                #                                                         units=self._temporal_gal_units)
                #             attention_output_list.append(attention_input)
                #
                #     graph_output = GCL.add_gc_layer(tf.reshape(tf.reduce_mean(attention_output_list[-1], axis=-2),
                #                                                [-1, self._num_node, attention_output_list[-1].get_shape()[-1].value]),
                #                                     K=self._gcn_k, laplacian_matrix=laplace_matrix[graph_index])
                #
                #     graph_outputs_list.append(tf.reshape(graph_output, [-1, 1, graph_output.get_shape()[-1].value]))

            if self._num_graph > 1:

                if self._graph_merge == 'gal':
                    # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                    _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(graph_outputs_list, axis=-2),
                                                            units=self._graph_merge_gal_units,
                                                            num_head=self._graph_merge_gal_num_heads)
                    dense_inputs = tf.reduce_mean(gal_output, axis=-2, keepdims=True)

                elif self._graph_merge == 'concat':

                    dense_inputs = tf.concat(graph_outputs_list, axis=-1)

            else:

                dense_inputs = graph_outputs_list[-1]

            dense_inputs = tf.reshape(dense_inputs, [-1, self._num_node, 1, dense_inputs.get_shape()[-1].value])

            if self._build_sd_regularization:
                sd_sim = tf.placeholder(tf.int32, [None, ])
                self._input['sd_sim'] = sd_sim.name
                sd_sim_features = tf.gather(dense_inputs, sd_sim, axis=1)
                sd_regularization_loss = tf.sqrt(tf.reduce_mean(tf.square(sd_sim_features - dense_inputs)))

            if self._build_transfer:
                self._output['feature_map'] = dense_inputs.name
                source_feature_map = tf.placeholder(tf.float32, dense_inputs.shape)
                self._input['similar_feature_map'] = source_feature_map.name
                # transfer_loss = tf.reduce_mean(tf.abs(source_feature_map - dense_inputs))
                transfer_loss = tf.sqrt(tf.reduce_mean(tf.square(source_feature_map - dense_inputs)))

            # dense_inputs = tf.layers.batch_normalization(dense_inputs, axis=-1, name='feature_map')
            #
            # batch_mean, batch_variance = tf.nn.moments(dense_inputs, [0, 1, 2], keep_dims=False)
            #
            # decay = 0.99
            # train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            # train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

            dense_inputs = keras.layers.BatchNormalization(axis=-1, name='feature_map')(dense_inputs)

            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_feature'] = external_input.name
                external_dense = tf.keras.layers.Dense(units=10)(external_input)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, tf.shape(dense_inputs)[1], tf.shape(dense_inputs)[2], 1])
                dense_inputs = tf.concat([dense_inputs, external_dense], axis=-1)

            conv1x1_output0 = tf.keras.layers.Conv2D(filters=self._num_filter_conv1x1,
                                                     kernel_size=[1, 1],
                                                     activation=tf.nn.tanh,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                     )(dense_inputs)

            conv1x1_output1 = tf.keras.layers.Conv2D(filters=self._num_filter_conv1x1,
                                                     kernel_size=[1, 1],
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                     activation=tf.nn.tanh,
                                                     )(conv1x1_output0)

            pre_output = tf.keras.layers.Conv2D(filters=1,
                                                kernel_size=[1, 1],
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                activation=self._output_activation,
                                                )(conv1x1_output1)

            prediction = tf.reshape(pre_output, [-1, self._num_node, 1], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')

            if self._build_sd_regularization:
                train_op = tf.train.AdamOptimizer(self._lr).minimize(loss_pre + sd_regularization_loss * 0.1,
                                                                     name='train_op')
            else:
                train_op = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            if self._build_transfer:
                transfer_loss = self._transfer_ratio * transfer_loss + (1 - self._transfer_ratio) * loss_pre
                transfer_op = tf.train.AdamOptimizer(self._lr).minimize(transfer_loss, name='transfer_op')
                self._output['transfer_loss'] = transfer_loss.name
                self._op['transfer_op'] = transfer_op.name

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_op.name

        super(AMultiGCLSTM, self).build(init_vars, max_to_keep)

    # Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self,
                       laplace_matrix,
                       closeness_feature=None,
                       period_feature=None,
                       trend_feature=None,
                       sd_sim=None,
                       target=None,
                       external_feature=None,
                       similar_feature_map=None):
        feed_dict = {
            'laplace_matrix': laplace_matrix,
        }
        if target is not None:
            feed_dict['target'] = target
        if similar_feature_map is not None:
            feed_dict['similar_feature_map'] = similar_feature_map
        if self._external_dim is not None and self._external_dim > 0:
            feed_dict['external_feature'] = external_feature
        if self._closeness_len is not None and self._closeness_len > 0:
            feed_dict['closeness_feature'] = closeness_feature
        if self._period_len is not None and self._period_len > 0:
            feed_dict['period_feature'] = period_feature
        if self._trend_len is not None and self._trend_len > 0:
            feed_dict['trend_feature'] = trend_feature
        if self._build_sd_regularization and sd_sim is not None:
            feed_dict['sd_sim'] = sd_sim
        return feed_dict
