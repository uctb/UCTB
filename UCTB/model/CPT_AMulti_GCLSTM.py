import numpy as np
import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import GCLSTMCell
from ..model_unit import GAL


class CPT_AMulti_GCLSTM(BaseModel):
    def __init__(self,
                 num_node,
                 num_graph,
                 external_dim,
                 C_T,
                 P_T,
                 T_T,
                 GCN_K=1,
                 GCN_layers=1,
                 GCLSTM_layers=1,
                 gal_units=32,
                 gal_num_heads=2,
                 num_hidden_units=64,
                 num_filter_conv1x1=32,
                 lr=5e-4,
                 code_version='QuickStart',
                 model_dir='model_dir',
                 GPU_DEVICE='0'):

        super(CPT_AMulti_GCLSTM, self).__init__(code_version=code_version, model_dir=model_dir, GPU_DEVICE=GPU_DEVICE)

        self._num_node = num_node
        self._gcn_k = GCN_K
        self._gcn_layer = GCN_layers
        self._gal_units = gal_units
        self._gal_num_heads = gal_num_heads
        self._gclstm_layers = GCLSTM_layers
        self._num_graph = num_graph
        self._external_dim = external_dim

        self._c_t = C_T
        self._p_t = P_T
        self._t_t = T_T
        self._num_hidden_unit = num_hidden_units
        self._num_filter_conv1x1 = num_filter_conv1x1
        self._lr = lr

    def build(self):
        with self._graph.as_default():

            temporal_features = []
            
            if self._c_t is not None and self._c_t > 0:
                closeness_feature = tf.placeholder(tf.float32, [None, 1, None, self._c_t], name='closeness_feature')
                self._input['closeness_feature'] = closeness_feature.name
                temporal_features.append([self._c_t, closeness_feature, 'closeness_feature'])

            if self._p_t is not None and self._p_t > 0:
                period_feature = tf.placeholder(tf.float32, [None, 1, None, self._p_t], name='period_feature')
                self._input['period_feature'] = period_feature.name
                temporal_features.append([self._p_t, period_feature, 'period_feature'])

            if self._t_t is not None and self._t_t > 0:
                trend_feature = tf.placeholder(tf.float32, [None, 1, None, self._t_t], name='trend_feature')
                self._input['trend_feature'] = trend_feature.name
                temporal_features.append([self._t_t, trend_feature, 'trend_feature'])

            if len(temporal_features) > 0:
                target = tf.placeholder(tf.float32, [None, None, 1], name='target')
                laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')
                self._input['target'] = target.name
                self._input['laplace_matrix'] = laplace_matrix.name
            else:
                raise ValueError('CT, PT, TT cannot all be zero')

            def dynamic_rnn(target_tensor, time_step, variable_scope_name):

                with tf.variable_scope(variable_scope_name, reuse=False):

                    outputs = []

                    if type(self._gcn_k) is list:
                        if len(self._gcn_k) != self._num_graph:
                            raise ValueError('Please provide K,L for each graph or set K,L to integer')
                        gc_lstm_cells = [
                            GCLSTMCell(self._gcn_k[graph_index], self._gcn_layer[graph_index], self._num_node,
                                       self._num_hidden_unit, state_is_tuple=True,
                                       initializer=tf.contrib.layers.xavier_initializer())
                            for _ in range(self._gclstm_layers)]
                    else:
                        gc_lstm_cells = [
                            GCLSTMCell(self._gcn_k, self._gcn_layer, self._num_node,
                                       self._num_hidden_unit, state_is_tuple=True,
                                       initializer=tf.contrib.layers.xavier_initializer())
                            for _ in range(self._gclstm_layers)]

                    for cell in gc_lstm_cells:
                        cell.laplacian_matrix = tf.transpose(laplace_matrix[graph_index])

                    cell_state_list = [cell.zero_state(tf.shape(target_tensor)[0], dtype=tf.float32)
                                       for cell in gc_lstm_cells]

                    for i in range(0, time_step):
                        output = target_tensor[:, 0, :, i:i + 1]
                        for cell_index in range(len(gc_lstm_cells)):
                            output, cell_state_list[cell_index] = gc_lstm_cells[cell_index](output,
                                                                                            cell_state_list[cell_index])
                        outputs.append(output)

                return outputs

            outputs_last_list = []

            for graph_index in range(self._num_graph):

                outputs_last = []

                for time_step, target_tensor, given_name in temporal_features:

                    outputs = dynamic_rnn(target_tensor, time_step, 'GCLSTM_%s_%s' % (graph_index, given_name))

                    outputs_last.append(tf.reshape(outputs[-1], [-1, 1, self._num_hidden_unit]))

                outputs_last_list.append(tf.concat(outputs_last, axis=-1))

            if self._num_graph > 1:
                # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(outputs_last_list, axis=-2),
                                                        units=self._gal_units, num_head=self._gal_num_heads)

                pre_input = tf.reshape(tf.reduce_mean(gal_output, axis=-2),
                                       [-1, self._num_node, 1, self._gal_units])
            else:
                pre_input = tf.reshape(outputs_last_list[-1],
                                       [-1, self._num_node, 1, self._num_hidden_unit * len(temporal_features)])

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

            prediction = tf.reshape(pre_output, [-1, self._num_node, 1], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_operation = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_operation.name

        super(CPT_AMulti_GCLSTM, self).build()

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
            feed_dict['external_input'] = external_feature
        if self._c_t is not None and self._c_t > 0:
            feed_dict['closeness_feature'] = closeness_feature
        if self._p_t is not None and self._p_t > 0:
            feed_dict['period_feature'] = period_feature
        if self._t_t is not None and self._t_t > 0:
            feed_dict['trend_feature'] = trend_feature
        return feed_dict

    # Step 2 : build the fit function using BaseModel._fit
    def fit(self,
            laplace_matrix,
            target,
            closeness_feature=None,
            period_feature=None,
            trend_feature=None,
            external_feature=None,
            batch_size=64, max_epoch=10000,
            validate_ratio=0.1,
            early_stop_method='t-test',
            early_stop_length=10,
            early_stop_patience=0.1):

        evaluate_loss_name = 'loss'

        feed_dict = self._get_feed_dict(closeness_feature=closeness_feature,
                                        period_feature=period_feature,
                                        trend_feature=trend_feature,
                                        laplace_matrix=laplace_matrix,
                                        target=target, external_feature=external_feature)

        return self._fit(feed_dict=feed_dict,
                         sequence_index=[e for e in feed_dict if e.endswith('_feature')][-1],
                         output_names=[evaluate_loss_name],
                         evaluate_loss_name=evaluate_loss_name,
                         op_names=['train_op'],
                         batch_size=batch_size,
                         max_epoch=max_epoch,
                         validate_ratio=validate_ratio,
                         early_stop_method=early_stop_method,
                         early_stop_length=early_stop_length,
                         early_stop_patience=early_stop_patience)

    def predict(self,
                laplace_matrix,
                closeness_feature=None,
                period_feature=None,
                trend_feature=None,
                external_feature=None,
                de_normalizer=None,
                cache_volume=64):

        feed_dict = self._get_feed_dict(closeness_feature=closeness_feature,
                                        period_feature=period_feature,
                                        trend_feature=trend_feature,
                                        laplace_matrix=laplace_matrix,
                                        external_feature=external_feature)

        output = self._predict(feed_dict=feed_dict, output_names=['prediction'], sequence_length=len(closeness_feature),
                               cache_volume=cache_volume)
        if de_normalizer is None:
            return output['prediction']
        else:
            return de_normalizer(output['prediction'])

    def evaluate(self, laplace_matrix, target, metrics,
                 closeness_feature=None, period_feature=None, trend_feature=None, external_feature=None,
                 de_normalizer=None, cache_volume=64, **kwargs):

        prediction = self.predict(closeness_feature=closeness_feature,
                                  laplace_matrix=laplace_matrix,
                                  period_feature=period_feature,
                                  trend_feature=trend_feature,
                                  external_feature=external_feature,
                                  de_normalizer=de_normalizer,
                                  cache_volume=cache_volume)

        if de_normalizer is not None:
            return [e(prediction=prediction, target=de_normalizer(target), **kwargs) for e in metrics]
        else:
            return [e(prediction=prediction, target=target, **kwargs) for e in metrics]