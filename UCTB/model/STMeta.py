import keras
import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import GAL, GCL
from ..model_unit import DCGRUCell
from ..model_unit import GCLSTMCell


class STMeta(BaseModel):
    """
        Args:
            num_node(int): Number of nodes in the graph, e.g. number of stations in NYC-Bike dataset.
            external_dim(int): Dimension of the external feature, e.g. temperature and wind are two dimension.
            closeness_len(int): The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history.
            period_len(int): The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history.
            trend_len(int): The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history.
            input_dim(int): The dimension of input features. 1 if "with_tpe" (data_loader parameters) = False, otherwise 0.
            num_graph(int): Number of graphs used in STMeta.
            gcn_k(int): The highest order of Chebyshev Polynomial approximation in GCN.
            gcn_layers(int): Number of GCN layers.
            gclstm_layers(int): Number of STRNN layers, it works on all modes of STMeta such as GCLSTM and DCRNN.
            num_hidden_units(int): Number of hidden units of RNN.
            num_dense_units(int): Number of dense units.
            graph_merge_gal_units(int): Number of units in GAL for merging different graph features.
                Only works when graph_merge='gal'
            graph_merge_gal_num_heads(int): Number of heads in GAL for merging different graph features.
                Only works when graph_merge='gal'
            temporal_merge_gal_units(int): Number of units in GAL for merging different temporal features.
                Only works when temporal_merge='gal'
            temporal_merge_gal_num_heads(int): Number of heads in GAL for merging different temporal features.
                Only works when temporal_merge='gal'
            st_method(str): must in ['GCLSTM', 'DCRNN', 'GRU', 'LSTM'], which refers to different
                spatial-temporal modeling methods.
                'GCLSTM': GCN for modeling spatial feature, LSTM for modeling temporal feature.
                'DCRNN': Diffusion Convolution for modeling spatial feature, GRU for modeling temporam frature.
                'GRU': Ignore the spatial, and model the temporal feature using GRU
                'LSTM': Ignore the spatial, and model the temporal feature using LSTM
            temporal_merge(str): must in ['gal', 'concat'], refers to different temporal merging methods,
                'gal': merge using GAL,
                'concat': merge by concat and dense
            graph_merge(str): must in ['gal', 'concat'], refers to different graph merging methods,
                'gal': merge using GAL,
                'concat': merge by concat and dense
            output_activation(function): activation function, e.g. tf.nn.tanh
            lr(float): Learning rate. Default: 1e-5
            code_version(str): Current version of this model code, which will be used as filename for saving the model
            model_dir(str): The directory to store model files. Default:'model_dir'.
            gpu_device(str): To specify the GPU to use. Default: '0'.
        """

    def __init__(self,
                 num_node,
                 external_dim,
                 closeness_len,
                 period_len,
                 trend_len,
                 input_dim=1,

                 # gcn parameters
                 num_graph=1,
                 gcn_k=1,
                 gcn_layers=1,
                 gclstm_layers=1,

                 # dense units
                 num_hidden_units=64,
                 # LSTM units
                 num_dense_units=32,

                 # merge parameters
                 graph_merge_gal_units=32,
                 graph_merge_gal_num_heads=2,
                 temporal_merge_gal_units=64,
                 temporal_merge_gal_num_heads=2,

                 # network structure parameters
                 st_method='GCLSTM',  # gclstm
                 temporal_merge='gal',  # gal
                 graph_merge='gal',  # concat

                 output_activation=tf.nn.sigmoid,

                 lr=1e-4,
                 code_version='STMeta-QuickStart',
                 model_dir='model_dir',
                 gpu_device='0', **kwargs):

        super(STMeta, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)

        self._num_node = num_node
        self._input_dim = input_dim
        print("self._input_dim",self._input_dim)
        self._gcn_k = gcn_k
        self._gcn_layer = gcn_layers
        self._graph_merge_gal_units = graph_merge_gal_units
        self._graph_merge_gal_num_heads = graph_merge_gal_num_heads
        self._temporal_merge_gal_units = temporal_merge_gal_units
        self._temporal_merge_gal_num_heads = temporal_merge_gal_num_heads
        self._gclstm_layers = gclstm_layers
        self._num_graph = num_graph
        self._external_dim = external_dim
        self._output_activation = output_activation

        self._st_method = st_method.upper()
        self._temporal_merge = temporal_merge
        self._graph_merge = graph_merge

        self._closeness_len = int(closeness_len)
        self._period_len = int(period_len)
        self._trend_len = int(trend_len)
        self._num_hidden_unit = num_hidden_units
        self._num_dense_units = num_dense_units
        self._lr = lr
    
    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():

            temporal_features = []

            if self._closeness_len is not None and self._closeness_len > 0:
                closeness_feature = tf.placeholder(tf.float32, [None, None, self._closeness_len, self._input_dim],
                                                   name='closeness_feature')
                self._input['closeness_feature'] = closeness_feature.name
                temporal_features.append([self._closeness_len, closeness_feature, 'closeness_feature'])

            if self._period_len is not None and self._period_len > 0:
                period_feature = tf.placeholder(tf.float32, [None, None, self._period_len, self._input_dim],
                                                name='period_feature')
                self._input['period_feature'] = period_feature.name
                temporal_features.append([self._period_len, period_feature, 'period_feature'])

            if self._trend_len is not None and self._trend_len > 0:
                trend_feature = tf.placeholder(tf.float32, [None, None, self._trend_len, self._input_dim],
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
                                [GCLSTMCell(units=self._num_hidden_unit, num_node=self._num_node,
                                            laplacian_matrix=laplace_matrix[graph_index],
                                            gcn_k=self._gcn_k, gcn_l=self._gcn_layer)
                                 for _ in range(self._gclstm_layers)])

                            outputs = tf.keras.layers.RNN(multi_layer_cell)(tf.reshape(target_tensor, [-1, time_step, self._input_dim]))

                            st_outputs = tf.reshape(outputs, [-1, 1, self._num_hidden_unit])

                        elif self._st_method == 'DCRNN':
                        # laplace_matrix will be diffusion_matrix when self._st_method == 'DCRNN'

                            encoding_cells = [DCGRUCell(self._num_hidden_unit, self._input_dim, self._num_graph,
                                             laplace_matrix,
                                             max_diffusion_step=self._gcn_k,
                                             num_node=self._num_node, name=str(graph_index) + given_name) for _ in range(self._gclstm_layers)]

                            encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)

                            inputs_unstack = tf.unstack(tf.reshape(target_tensor, [-1, self._num_node, time_step]),
                                                        axis=-1)

                            outputs, _ = \
                                tf.contrib.rnn.static_rnn(encoding_cells, inputs_unstack, dtype=tf.float32)

                            st_outputs = tf.reshape(outputs[-1], [-1, 1, self._num_hidden_unit])

                        elif self._st_method == 'GRU':

                            multi_layer_gru = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(units=self._num_hidden_unit) for _ in range(self._gclstm_layers)])
                            
                            outputs = tf.keras.layers.RNN(multi_layer_gru)(
                                tf.reshape(target_tensor, [-1, time_step, self._input_dim]))
                            st_outputs = tf.reshape(outputs, [-1, 1, self._num_hidden_unit])

                        elif self._st_method == 'LSTM':

                            multi_layer_gru = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(units=self._num_hidden_unit) for _ in range(self._gclstm_layers)])
                            
                            outputs = tf.keras.layers.RNN(multi_layer_gru)(
                                tf.reshape(target_tensor, [-1, time_step, self._input_dim]))
                            st_outputs = tf.reshape(outputs, [-1, 1, self._num_hidden_unit])

                        outputs_temporal.append(st_outputs)

                    if self._temporal_merge == 'concat':
                        
                        graph_outputs_list.append(tf.concat(outputs_temporal, axis=-1))

                    elif self._temporal_merge == 'gal':

                        _, gal_output = GAL.add_ga_layer_matrix(inputs=tf.concat(outputs_temporal, axis=-2),
                                                                units=self._temporal_merge_gal_units,
                                                                num_head=self._temporal_merge_gal_num_heads)

                        graph_outputs_list.append(tf.reduce_mean(gal_output, axis=-2, keepdims=True))

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

            dense_inputs = tf.keras.layers.BatchNormalization(axis=-1, name='feature_map')(dense_inputs)

            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_feature'] = external_input.name
                external_dense = tf.keras.layers.Dense(units=10)(external_input)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, tf.shape(dense_inputs)[1], tf.shape(dense_inputs)[2], 1])
                dense_inputs = tf.concat([dense_inputs, external_dense], axis=-1)

            dense_output0 = tf.keras.layers.Dense(units=self._num_dense_units,
                                                  activation=tf.nn.tanh,
                                                  use_bias=True,
                                                  kernel_initializer='glorot_uniform',
                                                  bias_initializer='zeros',
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                                  bias_regularizer=tf.keras.regularizers.l2(0.01)
                                                  )(dense_inputs)

            dense_output1 = tf.keras.layers.Dense(units=self._num_dense_units,
                                                  activation=tf.nn.tanh,
                                                  use_bias=True,
                                                  kernel_initializer='glorot_uniform',
                                                  bias_initializer='zeros',
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                                  bias_regularizer=tf.keras.regularizers.l2(0.01)
                                                  )(dense_output0)

            pre_output = tf.keras.layers.Dense(units=1,
                                               activation=tf.nn.tanh,
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform',
                                               bias_initializer='zeros',
                                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                               bias_regularizer=tf.keras.regularizers.l2(0.01)
                                               )(dense_output1)

            prediction = tf.reshape(pre_output, [-1, self._num_node, 1], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')

            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_op.name

        super(STMeta, self).build(init_vars, max_to_keep)

    # Define your '_get_feed_dict function‘, map your input to the tf-model
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
        if self._closeness_len is not None and self._closeness_len > 0:
            feed_dict['closeness_feature'] = closeness_feature
        if self._period_len is not None and self._period_len > 0:
            feed_dict['period_feature'] = period_feature
        if self._trend_len is not None and self._trend_len > 0:
            feed_dict['trend_feature'] = trend_feature
        return feed_dict
