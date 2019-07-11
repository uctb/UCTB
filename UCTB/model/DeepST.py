import os
import tensorflow as tf

from ..model_unit import BaseModel


class DeepST(BaseModel):
    def __init__(self,
                 closeness_len,
                 period_len,
                 trend_len,
                 num_channel,
                 width,
                 height,
                 external_dim,
                 lr,
                 code_version='QuickStart',
                 model_dir='model_dir',
                 gpu_device='0'):
        
        super(DeepST, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)
        
        self._num_channel = num_channel
        self._width = width
        self._height = height

        self._closeness_len = closeness_len
        self._period_len = period_len
        self._trend_len = trend_len

        self._external_dim = external_dim
        self._lr = lr

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
            h_c_1 = tf.layers.conv2d(inputs=c_conf, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)
            h_p_1 = tf.layers.conv2d(inputs=p_conf, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)
            h_t_1 = tf.layers.conv2d(inputs=t_conf, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)

            # First fusion
            h_2 = tf.layers.conv2d(tf.concat([h_c_1, h_p_1, h_t_1], axis=-1),
                                   filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)

            # Stack more convolutions
            middle_output = tf.layers.conv2d(h_2, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)
            x = tf.layers.conv2d(middle_output, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)

            # external dims
            if self._external_dim is not None and  self._external_dim > 0:
                external_input = tf.placeholder([None, self._external_dim])
                self._input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [1, 1, 1, -1]),
                                         [tf.shape(c_conf)[0], self._height, self._width, 1])
                x = tf.concat([x, external_dense])

            x = tf.layers.dense(x, units=1, name='prediction')

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
