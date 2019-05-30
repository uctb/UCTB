import numpy as np
import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import GCLSTMCell
from ..model_unit import GAL


class CPT_AMulti_GCLSTM_GAL(BaseModel):
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
                 pt_al_units=32,
                 pt_al_num_heads=2,
                 num_hidden_units=64,
                 num_filter_conv1x1=32,
                 lr=5e-4,
                 code_version='QuickStart',
                 model_dir='model_dir',
                 GPU_DEVICE='0'):

        super(CPT_AMulti_GCLSTM_GAL, self).__init__(code_version=code_version, model_dir=model_dir,
                                                    GPU_DEVICE=GPU_DEVICE)
        
        self._num_node = num_node
        self._gcn_k = GCN_K
        self._gcn_layer = GCN_layers
        self._gal_units = gal_units
        self._gal_num_heads = gal_num_heads
        self._pt_al_units = pt_al_units
        self._pt_al_num_heads = pt_al_num_heads
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

            closeness_feature = tf.placeholder(tf.float32, [None, 1, None, self._c_t], name='closeness_feature')
            target = tf.placeholder(tf.float32, [None, None, 1], name='target')
            laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')\

            self._input['closeness_feature'] = closeness_feature.name
            self._input['target'] = target.name
            self._input['laplace_matrix'] = laplace_matrix.name

            if self._c_t is not None and self._c_t > 0:

                batch_size = tf.shape(closeness_feature)[0]

                outputs_last_list = []

                for graph_index in range(self._num_graph):
                    with tf.variable_scope('gc_lstm_%s' % graph_index, reuse=False):
                        outputs_all = []

                        if type(self._gcn_k) is list:
                            if len(self._gcn_k) != self._num_graph:
                                raise ValueError('Please provide K,L for each graph or set K,L to integer')
                            gc_lstm_cells = [GCLSTMCell(self._gcn_k[graph_index], self._gcn_layer[graph_index], self._num_node,
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

                        cell_state_list = [cell.zero_state(batch_size, dtype=tf.float32) for cell in gc_lstm_cells]

                        for i in range(0, self._c_t):

                            output = closeness_feature[:, 0, :, i:i+1]

                            for cell_index in range(len(gc_lstm_cells)):

                                output, cell_state_list[cell_index] = gc_lstm_cells[cell_index](output, cell_state_list[cell_index])

                            outputs_all.append(output)

                    outputs_last_list.append(tf.reshape(outputs_all[-1], [-1, 1, self._num_hidden_unit]))

                if self._num_graph > 1:
                    # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                    _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(outputs_last_list, axis=-2),
                                                            units=self._gal_units, num_head=self._gal_num_heads)

                    pre_input = tf.reshape(tf.reduce_mean(gal_output, axis=-2),
                                        [-1, self._num_node, 1, self._gal_units])
                else:
                    pre_input = tf.reshape(outputs_last_list[-1], [-1, self._num_node, 1, self._num_hidden_unit])

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

                prediction = tf.reshape(pre_output, [batch_size, self._num_node, 1], name='prediction')

            else:
                prediction = []

            if self._p_t is not None and self._p_t > 0:
                period_feature = tf.placeholder(tf.float32, [None, self._p_t, None, self._c_t + 1],
                                                name='period_feature')
                self._input['period_feature'] = period_feature.name
                merge_item = tf.transpose(period_feature[:, :, :, -1:], perm=[0, 2, 1, 3])
                attention_feature = period_feature[:, :, :, :-1]
                attention_feature = tf.concat([closeness_feature, attention_feature], axis=1)
                attention_feature = tf.reshape(tf.transpose(attention_feature, perm=[0, 2, 1, 3]),
                                               [-1, 1+self._p_t, self._c_t])
                merge_weight = GAL.attention_merge_weight(attention_feature,
                                                          units=self._pt_al_units, num_head=self._pt_al_num_heads)
                merge_weight = tf.reshape(merge_weight, [-1, self._num_node, 1, self._p_t])
                prediction_period = tf.reshape(tf.matmul(merge_weight, merge_item), [batch_size, self._num_node, 1])

                prediction = tf.concat([prediction, prediction_period], axis=-1)

            if self._t_t is not None and self._t_t > 0:
                trend_feature = tf.placeholder(tf.float32, [None, self._t_t, None, self._c_t + 1],
                                               name='trend_feature')
                self._input['trend_feature'] = trend_feature.name
                merge_item = tf.transpose(trend_feature[:, :, :, -1:], perm=[0, 2, 1, 3])
                attention_feature = trend_feature[:, :, :, :-1]
                attention_feature = tf.concat([closeness_feature, attention_feature], axis=1)
                attention_feature = tf.reshape(tf.transpose(attention_feature, perm=[0, 2, 1, 3]),
                                               [-1, 1 + self._t_t, self._c_t])
                merge_weight = GAL.attention_merge_weight(attention_feature,
                                                          units=self._pt_al_units, num_head=self._pt_al_num_heads)
                merge_weight = tf.reshape(merge_weight, [-1, self._num_node, 1, self._t_t])
                prediction_trend = tf.reshape(tf.matmul(merge_weight, merge_item), [batch_size, self._num_node, 1])

                prediction = tf.concat([prediction, prediction_trend], axis=-1)

            if prediction.get_shape()[-1] > 1:
                merge_weight = tf.Variable(np.ones([1, prediction.get_shape()[-1].value, 1]) /
                                           prediction.get_shape()[-1].value, dtype=tf.float32)
                prediction = tf.matmul(prediction, tf.tile(merge_weight, [tf.shape(prediction)[0], 1, 1]))

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_operation = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_operation.name

        super(CPT_AMulti_GCLSTM_GAL, self).build()

    # Step 1 : Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self,
                       closeness_feature,
                       laplace_matrix,
                       period_feature=None,
                       trend_feature=None,
                       target=None,
                       external_feature=None):
        feed_dict = {
            'closeness_feature': closeness_feature,
            'laplace_matrix': laplace_matrix,
        }
        if target is not None:
            feed_dict['target'] = target
        if self._external_dim is not None and self._external_dim > 0:
            feed_dict['external_input'] = external_feature
        if self._p_t is not None and self._p_t > 0:
            feed_dict['period_feature'] = period_feature
        if self._t_t is not None and self._t_t > 0:
            feed_dict['trend_feature'] = trend_feature
        return feed_dict

    # Step 2 : build the fit function using BaseModel._fit
    def fit(self,
            closeness_feature,
            laplace_matrix,
            target,
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
                         sequence_index='closeness_feature',
                         output_names=[evaluate_loss_name],
                         evaluate_loss_name=evaluate_loss_name,
                         op_names=['train_op'],
                         batch_size=batch_size,
                         start_epoch=self._global_step,
                         max_epoch=max_epoch,
                         validate_ratio=validate_ratio,
                         early_stop_method=early_stop_method,
                         early_stop_length=early_stop_length,
                         early_stop_patience=early_stop_patience)

    def predict(self, closeness_feature,
                laplace_matrix,
                period_feature=None,
                trend_feature=None,
                external_feature=None,
                cache_volume=64):

        feed_dict = self._get_feed_dict(closeness_feature=closeness_feature,
                                        period_feature=period_feature,
                                        trend_feature=trend_feature,
                                        laplace_matrix=laplace_matrix,
                                        external_feature=external_feature)

        output = self._predict(feed_dict=feed_dict, output_names=['prediction'], sequence_length=len(closeness_feature),
                               cache_volume=cache_volume)

        return output['prediction']

    def evaluate(self, closeness_feature, laplace_matrix, target, metrics,
                 period_feature=None, trend_feature=None, external_feature=None, cache_volume=64, **kwargs):

        prediction = self.predict(closeness_feature=closeness_feature,
                                  laplace_matrix=laplace_matrix,
                                  period_feature=period_feature,
                                  trend_feature=trend_feature,
                                  external_feature=external_feature, cache_volume=cache_volume)

        return [e(prediction=prediction, target=target, **kwargs) for e in metrics]