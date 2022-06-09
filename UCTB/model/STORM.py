import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import GCLSTMCell
from ..model_unit import GCL

class STORM(BaseModel):
    """
    Spatio-Temporal Over-Demand Region Modeling
    """
    def __init__(self,
                # data params
                num_node,
                external_dim,
                closeness_len,
                period_len=0,
                trend_len=0,

                # gcn params
                num_graph=1,
                gcn_k=1,
                gcn_layers=1,
                gclstm_layers=1,
                gru_layers=1,

                # network params
                num_hidden_units=64,
                num_dense_units=32,
                output_activation=tf.nn.sigmoid,
                lr=1e-4,

                # loss weighted
                loss_w_node = 1,
                loss_w_tef = 1,
                
                # model params
                code_version='STORM',
                model_dir='model_dir',
                gpu_device='0', **kwargs):

        super(STORM, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)

        # data params
        self._num_node = num_node
        self._external_dim = external_dim
        self._closeness_len = int(closeness_len)
        self._period_len = int(period_len)
        self._trend_len = int(trend_len)

        # gcn params
        self._num_graph = num_graph
        self._gcn_k = gcn_k
        self._gcn_layer = gcn_layers
        self._gclstm_layers = gclstm_layers
        self._gru_layers = gru_layers

        # network params
        self._num_hidden_unit = num_hidden_units
        self._num_dense_units = num_dense_units
        self._output_activation = output_activation
        self._lr = lr
        
        # loss weighted
        self._loss_w_node = loss_w_node
        self._loss_w_tef = loss_w_tef


    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():

            # features
            spatial_external_feature = tf.placeholder(tf.float32, [None, self._num_node], name='spatial_external_feature')
            self._input['spatial_external_feature'] = spatial_external_feature.name

            temporal_traffic_features = []
            temporal_external_features = []

            if self._closeness_len is not None and self._closeness_len > 0:
                # traffic
                closeness_traffic_feature = tf.placeholder(tf.float32, [None, None, self._closeness_len, 1],
                                                   name='closeness_traffic_feature')
                self._input['closeness_traffic_feature'] = closeness_traffic_feature.name
                temporal_traffic_features.append([self._closeness_len, closeness_traffic_feature, 'closeness_traffic_feature'])
                # temporal external
                closeness_external_feature = tf.placeholder(tf.float32, [None, None, self._closeness_len, 1],
                                                   name='closeness_external_feature')
                self._input['closeness_external_feature'] = closeness_external_feature.name
                temporal_external_features.append([self._closeness_len, closeness_external_feature, 'closeness_external_feature'])

            if self._period_len is not None and self._period_len > 0:
                # traffic
                period_feature = tf.placeholder(tf.float32, [None, None, self._period_len, 1],
                                                name='period_traffic_feature')
                self._input['period_traffic_feature'] = period_feature.name
                temporal_traffic_features.append([self._period_len, period_feature, 'period_traffic_feature'])
                # temporal external
                period_external_feature = tf.placeholder(tf.float32, [None, None, self._period_len, 1], name='period_external_feature')
                self._input['period_external_feature'] = period_external_feature.name
                temporal_external_features.append([self._period_len, period_external_feature, 'period_external_feature'])

            if self._trend_len is not None and self._trend_len > 0:
                # traffic
                trend_feature = tf.placeholder(tf.float32, [None, None, self._trend_len, 1],
                                               name='trend_traffic_feature')
                self._input['trend_traffic_feature'] = trend_feature.name
                temporal_traffic_features.append([self._trend_len, trend_feature, 'trend_traffic_feature'])
                # temporal external
                trend_external_feature = tf.placeholder(tf.float32, [None, None, self._closeness_len, 1],
                                                   name='trend_external_feature')
                self._input['trend_external_feature'] = trend_external_feature.name
                temporal_external_features.append([self._closeness_len, trend_external_feature, 'trend_external_feature'])

            if len(temporal_traffic_features) > 0:
                # laplace matrix
                laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')
                self._input['laplace_matrix'] = laplace_matrix.name
                # traffic
                target_traffic = tf.placeholder(tf.float32, [None, None, 1], name='target_traffic')
                self._input['target_traffic'] = target_traffic.name
                # temporal external
                target_external = tf.placeholder(tf.float32, [None, None, 1], name='target_external')
                self._input['target_external'] = target_external.name
            else:
                raise ValueError('closeness_len, period_len, trend_len cannot all be zero')

            graph_outputs_list = []
            # use gcn to extract spatial_external_feature
            sef_outputs_list = []

            # multi-graph to train traffic feature
            for graph_index in range(self._num_graph):

                # spatial-temporal features
                outputs_traffic_temporal = []

                for time_step, traffic_tensor, given_name in temporal_traffic_features:
                    # GCLSTM
                    multi_gclstm_cell = tf.keras.layers.StackedRNNCells(
                        [GCLSTMCell(units=self._num_hidden_unit, num_nodes=self._num_node,
                                    laplacian_matrix=laplace_matrix[graph_index],
                                    gcn_k=self._gcn_k, gcn_l=self._gcn_layer)
                         for _ in range(self._gclstm_layers)])
                    # traffic_outputs shape is [time_slot*node_num, time_step, 1] -> [time_slot*node_num, hidden_unit]
                    traffic_outputs = tf.keras.layers.RNN(multi_gclstm_cell)(tf.reshape(traffic_tensor, [-1, time_step, 1]))
                    st_outputs = tf.reshape(traffic_outputs, [-1, 1, self._num_hidden_unit])
                    outputs_traffic_temporal.append(st_outputs)

                # temporal merge: concat
                graph_outputs_list.append(tf.concat(outputs_traffic_temporal, axis=-1))

                sef_gcl_output = GCL.add_multi_gc_layers(tf.reshape(spatial_external_feature, [-1, self._num_node, 1]), graph_id=graph_index,
                                                    gcn_k=self._gcn_k, gcn_l=self._gcn_layer, output_size=1,
                                                    laplacian_matrix=laplace_matrix[graph_index],
                                                    activation=tf.nn.tanh)
                sef_outputs_list.append(sef_gcl_output)

            if self._num_graph > 1:
                # graph merge: concat, traffic_inputs shape is [node_num*time_slot, 1, hidden_unit*2*graph_num]
                traffic_inputs = tf.concat(graph_outputs_list, axis=-1)
                # sef_inputs shape is [time_slot, node_num, graph_num]
                sef_inputs = tf.concat(sef_outputs_list, axis=1)

            else:
                traffic_inputs = graph_outputs_list[-1]
            # traffic_inputs shape is [time_slot, node_num, 1, hidden_unit*2*graph_num]
            traffic_inputs = tf.reshape(traffic_inputs, [-1, self._num_node, 1, traffic_inputs.get_shape()[-1].value])
            sef_inputs = tf.reshape(sef_inputs, [-1, self._num_node*self._num_graph])
            # print('========traffic_inputs shape ========')
            # print(tf.shape(traffic_inputs))

            # GRU to train temporal external feature
            outputs_external_temporal = []
            for time_step, external_tensor, given_name in temporal_external_features:
                # GRU
                multi_layer_gru = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(units=self._num_hidden_unit) for _ in range(self._gclstm_layers)])
                # outputs shape is [ext_dim*time_slot, time_step, 1] --> [ext_dim*time_slot, hidden_unit]
                outputs = tf.keras.layers.RNN(multi_layer_gru)(tf.reshape(external_tensor, [-1, time_step, 1]))
                outputs_external_temporal.append(tf.reshape(outputs, [-1, 1, self._num_hidden_unit]))
            # external_inputs shape is [ext_dim*time_slot, hidden_unit*2]
            external_inputs = tf.concat(outputs_external_temporal, axis=-1)
            # external_inputs shape is [time_slot, ext_dim, 1, hidden_unit*2]
            external_inputs = tf.reshape(external_inputs, [-1, self._external_dim, 1, external_inputs.get_shape()[-1].value])
            # print('========external_inputs shape ========')
            # print(tf.shape(external_inputs))
            # external_inputs shape is [time_slot, ext_dim, 1, hidden_unit*2*graph_num]
            external_inputs = tf.tile(external_inputs, [1, 1, 1, self._num_graph])
            # print('========after reshape and tile, the shape of external_inputs is ========')
            # print(tf.shape(external_inputs))

            # Todo multi-task learning
            # multi_task_flow shape is [time_slot, num_node+ext_dim, 1, hidden_unit*2*graph_num]
            multi_task_flow = tf.concat([traffic_inputs, external_inputs], axis=1)
            # multi_task_flow shape is [time_slot, hidden_unit*2*graph_num, 1, num_node+ext_dim]
            multi_task_flow = tf.transpose(multi_task_flow, [0, 3, 2, 1])
            multi_task_flow = tf.keras.layers.BatchNormalization(axis=-1, name='feature_map')(multi_task_flow)
            # print('========multi_task_flow shape ========')
            # print(tf.shape(multi_task_flow))

            # traffic output
            output_traffic_gru = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(units=self._num_hidden_unit) for _ in range(self._gru_layers)])
            # gru_output_traffic shape is [time_slot, node_num*hidden_unit*2*graph_num]
            gru_output_traffic = tf.keras.layers.RNN(output_traffic_gru)(tf.reshape(multi_task_flow, [-1, multi_task_flow.get_shape()[1].value, multi_task_flow.get_shape()[-1].value]))
            # fusion_external, dense_traffic_inputs shape is [time_slot, hidden_unit+sef]
            dense_traffic_inputs = tf.keras.layers.BatchNormalization(axis=-1)(tf.concat([gru_output_traffic, sef_inputs], axis=1))
            dense_traffic_output0 = tf.keras.layers.Dense(units=self._num_dense_units,
                                      activation=tf.nn.tanh,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      bias_regularizer=tf.keras.regularizers.l2(0.01)
                                      )(dense_traffic_inputs)
            dense_traffic_output1 = tf.keras.layers.Dense(units=self._num_dense_units,
                                      activation=tf.nn.tanh,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      bias_regularizer=tf.keras.regularizers.l2(0.01)
                                      )(dense_traffic_output0)
            pre_traffic_output = tf.keras.layers.Dense(units=self._num_node,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   bias_regularizer=tf.keras.regularizers.l2(0.01)
                                   )(dense_traffic_output1)
            # print('========pre_traffic_output shape ========')
            # print(tf.shape(pre_traffic_output))
            traffic_prediction = tf.reshape(pre_traffic_output, [-1, self._num_node, 1], name='prediction')

            # external output
            output_external_gru = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(units=self._num_hidden_unit) for _ in range(self._gru_layers)])
            # gru_output_external shape is [time_slot*hidden_unit*2*graph_num, node_num]
            gru_output_external = tf.keras.layers.RNN(output_external_gru)(tf.reshape(multi_task_flow, [-1, multi_task_flow.get_shape()[1].value, multi_task_flow.get_shape()[-1].value]))
            dense_external_inputs = tf.keras.layers.BatchNormalization(axis=-1)(tf.concat([gru_output_external, sef_inputs], axis=1))
            dense_external_output0 = tf.keras.layers.Dense(units=self._num_dense_units,
                                      activation=tf.nn.tanh,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      bias_regularizer=tf.keras.regularizers.l2(0.01)
                                      )(dense_external_inputs)
            dense_external_output1 = tf.keras.layers.Dense(units=self._num_dense_units,
                                      activation=tf.nn.tanh,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      bias_regularizer=tf.keras.regularizers.l2(0.01)
                                      )(dense_external_output0)
            pre_external_output = tf.keras.layers.Dense(units=self._external_dim,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   bias_regularizer=tf.keras.regularizers.l2(0.01)
                                   )(dense_external_output1)
            external_prediction = tf.reshape(pre_external_output, [-1, self._external_dim, 1], name='external_prediction')
            # loss define
            traffic_loss = tf.multiply(self._loss_w_node, tf.reduce_mean(tf.square(target_traffic - traffic_prediction)))
            external_loss = tf.multiply(self._loss_w_tef, tf.reduce_mean(tf.square(target_external - external_prediction)))
            loss_all = tf.add(traffic_loss, external_loss, name='loss')
            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss_all, name='train_op')
            # output
            self._output['prediction'] = traffic_prediction.name
            self._output['external_prediction'] = external_prediction.name
            self._output['loss'] = loss_all.name
            self._op['train_op'] = train_op.name

        super(STORM, self).build(init_vars, max_to_keep)

    # Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self,
                        laplace_matrix,
                        closeness_traffic_feature=None,
                        period_traffic_feature=None,
                        trend_traffic_feature=None,
                        target_traffic=None,
                        closeness_external_feature=None,
                        period_external_feature = None,
                        trend_external_feature = None,
                        target_external = None,
                        spatial_external_feature=None
                        ):
        feed_dict = {
            'laplace_matrix': laplace_matrix,
        }
        if target_traffic is not None:
            feed_dict['target_traffic'] = target_traffic
            feed_dict['target_external'] = target_external
        if spatial_external_feature is not None:
            feed_dict['spatial_external_feature'] = spatial_external_feature
        if self._closeness_len is not None and self._closeness_len > 0:
            feed_dict['closeness_traffic_feature'] = closeness_traffic_feature
            feed_dict['closeness_external_feature'] = closeness_external_feature
        if self._period_len is not None and self._period_len > 0:
            feed_dict['period_traffic_feature'] = period_traffic_feature
            feed_dict['period_external_feature'] = period_external_feature
        if self._trend_len is not None and self._trend_len > 0:
            feed_dict['trend_traffic_feature'] = trend_traffic_feature
            feed_dict['trend_external_feature'] = trend_external_feature
        return feed_dict
