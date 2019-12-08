import os
import tensorflow as tf

from ..model_unit import BaseModel


class DeepST(BaseModel):
    """Deep learning-based prediction model for Spatial-Temporal data (DeepST)

    DeepST is composed of three components: 1) temporal dependent
    instances: describing temporal closeness, period and seasonal
    trend; 2) convolutional neural networks: capturing near and
    far spatial dependencies; 3) early and late fusions: fusing
    similar and different domains' data.

    Reference:
        - `DNN-Based Prediction Model for Spatial-Temporal Data (Junbo Zhang et al., 2016)
          <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/DeepST-SIGSPATIAL2016.pdf>`_.

    Args:
        closeness_len (int): The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history.
        period_len (int): The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history.
        trend_len (int): The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history.
        width (int): The width of grid data.
        height (int): The height of grid data.
        externai_dim (int): Number of dimensions of external data.
        kernel_size (int): Kernel size in Convolutional neural networks. Default: 3
        num_conv_filters (int):  the Number of filters in the convolution. Default: 64
        lr (float): Learning rate. Default: 1e-5
        code_version (str): Current version of this model code.
        model_dir (str): The directory to store model files. Default:'model_dir'
        gpu_device (str): To specify the GPU to use. Default: '0'
    """
    def __init__(self,
                 closeness_len,
                 period_len,
                 trend_len,
                 width,
                 height,
                 external_dim,
                 kernel_size=3,
                 num_conv_filters=64,
                 lr=1e-5,
                 code_version='QuickStart-DeepST',
                 model_dir='model_dir',
                 gpu_device='0'):
        
        super(DeepST, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)

        self._width = width
        self._height = height

        self._closeness_len = closeness_len
        self._period_len = period_len
        self._trend_len = trend_len

        self._external_dim = external_dim
        self._lr = lr
        self._kernel_size = kernel_size
        self._num_conv_filters = num_conv_filters

        self._graph = tf.Graph()
        self._GPU_DEVICE = gpu_device

        self._input = {}
        self._output = {}
        self._op = {}
        self._variable_init = None
        self._saver = None
        self._model_dir = model_dir

        # GPU Config
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU_DEVICE
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True

        self._session = tf.Session(graph=self._graph, config=self._config)
    
    def build(self):
        with self._graph.as_default():

            c_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._closeness_len], name='c')
            p_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._period_len], name='p')
            t_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._trend_len], name='t')

            target = tf.placeholder(tf.float32, [None, self._height, self._width, 1], name='target')
            
            self._input['closeness_feature'] = c_conf.name
            self._input['period_feature'] = p_conf.name
            self._input['trend_feature'] = t_conf.name
            self._input['target'] = target.name

            # First convolution
            h_c_1 = tf.layers.conv2d(inputs=c_conf, filters=self._num_conv_filters,
                                     kernel_size=[self._kernel_size, self._kernel_size], padding='SAME', use_bias=True)
            h_p_1 = tf.layers.conv2d(inputs=c_conf, filters=self._num_conv_filters,
                                     kernel_size=[self._kernel_size, self._kernel_size], padding='SAME', use_bias=True)
            h_t_1 = tf.layers.conv2d(inputs=c_conf, filters=self._num_conv_filters,
                                     kernel_size=[self._kernel_size, self._kernel_size], padding='SAME', use_bias=True)

            # First fusion
            h_2 = tf.layers.conv2d(tf.concat([h_c_1, h_p_1, h_t_1], axis=-1),
                                   filters=self._num_conv_filters, kernel_size=[self._kernel_size, self._kernel_size],
                                   padding='SAME', use_bias=True)

            # Stack more convolutions
            middle_output = tf.layers.conv2d(h_2, filters=self._num_conv_filters,
                                             kernel_size=[self._kernel_size, self._kernel_size],
                                             padding='SAME', use_bias=True)
            x = tf.layers.conv2d(middle_output, filters=self._num_conv_filters,
                                 kernel_size=[self._kernel_size, self._kernel_size], padding='SAME', use_bias=True)

            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_feature'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]), [1, self._height, self._width, 1])
                x = tf.concat([x, external_dense], axis=-1)

            x = tf.layers.dense(x, units=1, name='prediction', activation=tf.nn.sigmoid)

            loss = tf.sqrt(tf.reduce_mean(tf.square(x - target)), name='loss')
            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss)

            self._output['prediction'] = x.name
            self._output['loss'] = loss.name

            self._op['train_op'] = train_op.name

            self._variable_init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        super(DeepST, self).build()

    # Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self, closeness_feature=None, period_feature=None, trend_feature=None,
                       target=None, external_feature=None):
        '''
        The method to get feet dict for tensorflow model.

        Users may modify this method according to the format of input.

        Args:
            closeness_feature (np.ndarray or ``None``): The closeness history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, closeness_len].
            period_feature (np.ndarray or ``None``): The period history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, period_len].
            trend_feature (np.ndarray or ``None``): The trend history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, trend_len].
            target (np.ndarray or ``None``): The target value data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, 1].
            external_feature (np.ndarray or ``None``): The external feature data.
                If type is np.ndaaray, its shape is [time_slot_num, feature_num].
        '''
        feed_dict = {}
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
