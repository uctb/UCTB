import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit.GraphModelLayers import GCL


class ST_MGCN(BaseModel):
    """

    References:
        - `Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting (Geng Xu, et al., 2019)
          <http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf>`_.
        - `A PyTorch implementation of the ST-MGCN model  (shawnwang-tech)
          <https://github.com/shawnwang-tech/ST-MGCN-pytorch>`_.

    Args:
        T(int): Input sequence length
        input_dim(int): Input feature dimension
        num_graph(int): Number of graphs used in the model.
        gcl_k(int): The highest order of Chebyshev Polynomial approximation in GCN.
        gcl_l(int): Number of GCN layers.
        lstm_units(int): Number of hidden units of RNN.
        lstm_layers(int): Number of LSTM layers.
        lr(float): Learning rate
        external_dim(int): Dimension of the external feature, e.g. temperature and wind are two dimension.
        code_version(str): Current version of this model code, which will be used as filename for saving the model
        model_dir(str): The directory to store model files. Default:'model_dir'.
        gpu_device(str): To specify the GPU to use. Default: '0'.
    """
    def __init__(self,
                 T,
                 input_dim,
                 num_graph,
                 gcl_k,
                 gcl_l,
                 lstm_units,
                 lstm_layers,
                 lr,
                 external_dim,
                 code_version,
                 model_dir,
                 gpu_device):

        super(ST_MGCN, self).__init__(code_version=code_version,
                                      model_dir=model_dir,
                                      gpu_device=gpu_device)

        self._T = T
        self._input_dim = input_dim
        self._num_graph = num_graph
        self._gcl_k = gcl_k
        self._gcl_l = gcl_l
        self._lstm_units = lstm_units
        self._lstm_layers = lstm_layers
        self._lr = lr
        self._external_dim = external_dim

    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():
            # [batch, T, num_stations, input_dim]
            traffic_flow = tf.placeholder(tf.float32, [None, self._T, None, self._input_dim])
            laplacian_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None])
            target = tf.placeholder(tf.float32, [None, None, 1])

            self._input['traffic_flow'] = traffic_flow.name
            self._input['laplace_matrix'] = laplacian_matrix.name
            self._input['target'] = target.name

            station_number = tf.shape(traffic_flow)[-2]

            outputs = []

            for graph_index in range(self._num_graph):
                with tf.variable_scope('CGRNN_Graph%s' % graph_index, reuse=False):
                    f_k_g = GCL.add_multi_gc_layers(tf.reshape(traffic_flow, [-1, station_number, self._input_dim]),
                                                    gcn_k=1, gcn_l=1, output_size=self._input_dim,
                                                    laplacian_matrix=laplacian_matrix[graph_index],
                                                    activation=tf.nn.tanh)

                    f_k_g = tf.reshape(f_k_g, [-1, self._T, station_number, self._input_dim])

                    x_hat = tf.concat([f_k_g, traffic_flow], axis=-1)

                    z = tf.reduce_mean(x_hat, axis=-2, keepdims=True)

                    s = tf.layers.dense(tf.layers.dense(z, units=4, use_bias=False, activation=tf.nn.relu),
                                        units=1, use_bias=False, activation=tf.nn.sigmoid)

                    x_rnn = tf.multiply(traffic_flow, tf.tile(s, [1, 1, station_number, self._input_dim]))

                    x_rnn = tf.reshape(tf.transpose(x_rnn, perm=[0, 2, 1, 3]), [-1, self._T, self._input_dim])

                    for lstm_layer_index in range(self._lstm_layers):
                        x_rnn = tf.keras.layers.LSTM(units=self._lstm_units,
                                                     activation='tanh',
                                                     dropout=0.1,
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                     return_sequences=True if lstm_layer_index<self._lstm_layers-1
                                                                      else False)\
                                                    (x_rnn)

                    H = tf.reshape(x_rnn, [-1, station_number, self._lstm_units])

                    outputs.append(H)

            outputs = tf.reduce_sum(outputs, axis=0)

            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_feature'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 10]),
                                         [1, tf.shape(outputs)[-2], 1])
                outputs = tf.concat([outputs, external_dense], axis=-1)

            prediction = tf.layers.dense(outputs, units=1)

            loss = tf.sqrt(tf.reduce_mean(tf.square(prediction - target)))

            train_operation = tf.train.AdamOptimizer(self._lr).minimize(loss, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss.name

            # record train operation
            self._op['train_op'] = train_operation.name

            super(ST_MGCN, self).build(init_vars=init_vars, max_to_keep=max_to_keep)

    # Step 1 : Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self, traffic_flow, laplace_matrix, target=None, external_feature=None):
        feed_dict = {
            'traffic_flow': traffic_flow,
            'laplace_matrix': laplace_matrix,
        }
        if target is not None:
            feed_dict['target'] = target
        if external_feature is not None:
            feed_dict['external_feature'] = external_feature
        return feed_dict
