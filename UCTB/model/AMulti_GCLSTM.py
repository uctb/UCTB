import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import GCLSTMCell
from ..model_unit import GAL, GCL


class AMulti_GCLSTM(BaseModel):
    def __init__(self,
                 num_node,
                 external_dim,
                 closeness_len,
                 period_len,
                 trend_len,

                 # gcn parameters
                 num_graph=1,
                 gcn_k=(1, ),
                 gcn_layers=(1, ),
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
                 st_method='gal_gcn',          # gclstm
                 temporal_merge='concat',     # gal
                 graph_merge='concat',        # concat

                 # Transfer learning
                 build_transfer=False,

                 lr=1e-4,
                 code_version='QuickStart',
                 model_dir='model_dir',
                 gpu_device='0', **kwargs):

        super(AMulti_GCLSTM, self).__init__(code_version=code_version, model_dir=model_dir, GPU_DEVICE=gpu_device)

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

        self._st_method = st_method
        self._temporal_merge = temporal_merge
        self._graph_merge = graph_merge

        self._build_transfer = build_transfer

        self._tpe_dim = tpe_dim
        if st_method == 'gal_gcn':
            assert self._tpe_dim

        self._c_t = closeness_len
        self._p_t = period_len
        self._t_t = trend_len
        self._num_hidden_unit = num_hidden_units
        self._num_filter_conv1x1 = num_filter_conv1x1
        self._lr = lr
    
    def dynamic_rnn(self, temporal_data, time_length, graph_index, laplace_matrix, variable_scope_name):
        with self._graph.as_default():
            with tf.variable_scope(variable_scope_name, reuse=False):
                outputs = []
                if hasattr(self._gcn_k, '__len__'):
                    if len(self._gcn_k) != self._num_graph:
                        raise ValueError('Please provide K,L for each graph or set K,L to integer')
                    gc_lstm_cells = [
                        GCLSTMCell(self._gcn_k[graph_index], self._gcn_layer[graph_index], self._num_node,
                                   self._num_hidden_unit, state_is_tuple=True,
                                   initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                        for _ in range(self._gclstm_layers)]
                else:
                    gc_lstm_cells = [
                        GCLSTMCell(self._gcn_k, self._gcn_layer, self._num_node,
                                   self._num_hidden_unit, state_is_tuple=True,
                                   initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                        for _ in range(self._gclstm_layers)]
                for cell in gc_lstm_cells:
                    cell.laplacian_matrix = tf.transpose(laplace_matrix[graph_index])
                cell_state_list = [cell.zero_state(tf.shape(temporal_data)[0], dtype=tf.float32)
                                   for cell in gc_lstm_cells]
                for i in range(0, time_length):
                    output = temporal_data[:, :, i:i + 1, 0]
                    for cell_index in range(len(gc_lstm_cells)):
                        output, cell_state_list[cell_index] = gc_lstm_cells[cell_index](output,
                                                                                        cell_state_list[cell_index])
                    outputs.append(output)
        return outputs
    
    def build(self):
        with self._graph.as_default():

            temporal_features = []

            if self._c_t is not None and self._c_t > 0:
                if self._st_method == 'gclstm':
                    closeness_feature = tf.placeholder(tf.float32, [None, None, self._c_t, 1],
                                                       name='closeness_feature')
                elif self._st_method == 'gal_gcn':
                    closeness_feature = tf.placeholder(tf.float32, [None, None, self._c_t, 1 + self._tpe_dim],
                                                       name='closeness_feature')
                self._input['closeness_feature'] = closeness_feature.name
                temporal_features.append([self._c_t, closeness_feature, 'closeness_feature'])

            if self._p_t is not None and self._p_t > 0:
                if self._st_method == 'gclstm':
                    period_feature = tf.placeholder(tf.float32, [None, None, self._p_t, 1],
                                                    name='period_feature')
                elif self._st_method == 'gal_gcn':
                    period_feature = tf.placeholder(tf.float32, [None, None, self._p_t, 1 + self._tpe_dim],
                                                    name='period_feature')
                self._input['period_feature'] = period_feature.name
                temporal_features.append([self._p_t, period_feature, 'period_feature'])

            if self._t_t is not None and self._t_t > 0:
                if self._st_method == 'gclstm':
                    trend_feature = tf.placeholder(tf.float32, [None, None, self._t_t, 1],
                                                   name='trend_feature')
                elif self._st_method == 'gal_gcn':
                    trend_feature = tf.placeholder(tf.float32, [None, None, self._t_t, 1 + self._tpe_dim],
                                                   name='trend_feature')
                self._input['trend_feature'] = trend_feature.name
                temporal_features.append([self._t_t, trend_feature, 'trend_feature'])

            if len(temporal_features) > 0:
                target = tf.placeholder(tf.float32, [None, None, 1], name='target')
                laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')
                self._input['target'] = target.name
                self._input['laplace_matrix'] = laplace_matrix.name
            else:
                raise ValueError('closeness_len, period_len, trend_len cannot all be zero')

            graph_outputs_list = []

            for graph_index in range(self._num_graph):

                if self._st_method == 'gclstm':

                    outputs_temporal = []

                    for time_step, target_tensor, given_name in temporal_features:

                        outputs = self.dynamic_rnn(temporal_data=target_tensor,
                                                   time_length=time_step,
                                                   graph_index=graph_index,
                                                   laplace_matrix=laplace_matrix,
                                                   variable_scope_name='GCLSTM_%s_%s' % (graph_index, given_name))

                        st_outputs = tf.reshape(outputs[-1], [-1, 1, self._num_hidden_unit])

                        outputs_temporal.append(st_outputs)

                    if self._temporal_merge == 'concat':
                        
                        graph_outputs_list.append(tf.concat(outputs_temporal, axis=-1))

                    elif self._temporal_merge == 'gal':

                        _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(outputs_temporal, axis=-2),
                                                                units=self._temporal_merge_gal_units,
                                                                num_head=self._temporal_merge_gal_num_heads)

                        graph_outputs_list.append(tf.reduce_mean(gal_output, axis=-2, keepdims=True))

                elif self._st_method == 'gal_gcn':

                    attention_input = tf.reshape(tf.concat([e[1] for e in temporal_features], axis=-2),
                                                 [-1, sum([e[0] for e in temporal_features]), 1 + self._tpe_dim])
                    attention_output_list = []
                    for loop_index in range(self._temporal_gal_layers):
                        with tf.variable_scope('res_temporal_gal_%s' % loop_index, reuse=False):
                            attention_input = GAL.add_residual_ga_layer(attention_input,
                                                                        num_head=self._temporal_gal_num_heads,
                                                                        units=self._temporal_gal_units)
                            attention_output_list.append(attention_input)

                    graph_output = GCL.add_gc_layer(tf.reshape(tf.reduce_mean(attention_output_list[-1], axis=-2),
                                                               [-1, self._num_node, attention_output_list[-1].get_shape()[-1].value]),
                                                    K=self._gcn_k, laplacian_matrix=laplace_matrix[graph_index])

                    graph_outputs_list.append(tf.reshape(graph_output, [-1, 1, graph_output.get_shape()[-1].value]))

            if self._num_graph > 1:

                if self._graph_merge == 'gal':
                    # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                    _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(graph_outputs_list, axis=-2),
                                                            units=self._graph_merge_gal_units, num_head=self._graph_merge_gal_num_heads)
                    dense_inputs = tf.reduce_mean(gal_output, axis=-2, keepdims=True)

                elif self._graph_merge == 'concat':

                    dense_inputs = tf.concat(graph_outputs_list, axis=-1)

            else:

                dense_inputs = graph_outputs_list[-1]

            dense_inputs = tf.reshape(dense_inputs, [-1, self._num_node, 1, dense_inputs.get_shape()[-1].value])

            dense_inputs = tf.layers.batch_normalization(dense_inputs, axis=-1, name='feature_map')

            if self._build_transfer:
                self._output['feature_map'] = dense_inputs.name
                source_feature_map = tf.placeholder(tf.float32, dense_inputs.shape)
                self._input['similar_feature_map'] = source_feature_map.name
                transfer_loss = tf.reduce_mean(tf.abs(source_feature_map - dense_inputs))

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
                                                     )(conv1x1_output0)

            pre_output = tf.keras.layers.Conv2D(filters=1,
                                                kernel_size=[1, 1],
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                                                )(conv1x1_output1)

            prediction = tf.reshape(pre_output, [-1, self._num_node, 1], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            if self._build_transfer:
                transfer_loss = 0.5 * transfer_loss + 0.5 * loss_pre
                transfer_op = tf.train.AdamOptimizer(self._lr).minimize(transfer_loss, name='transfer_op')
                self._output['transfer_loss'] = transfer_loss.name
                self._op['transfer_op'] = transfer_op.name

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_op.name

        super(AMulti_GCLSTM, self).build()

    # Step 1 : Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self,
                       laplace_matrix,
                       closeness_feature=None,
                       period_feature=None,
                       trend_feature=None,
                       target=None,
                       external_feature=None):
        feed_dict = {
            'laplace_matrix': laplace_matrix,
        }
        if target is not None:
            feed_dict['target'] = target
        if self._external_dim is not None and self._external_dim > 0:
            feed_dict['external_feature'] = external_feature
        if self._c_t is not None and self._c_t > 0:
            feed_dict['closeness_feature'] = closeness_feature
        if self._p_t is not None and self._p_t > 0:
            feed_dict['period_feature'] = period_feature
        if self._t_t is not None and self._t_t > 0:
            feed_dict['trend_feature'] = trend_feature
        return feed_dict
