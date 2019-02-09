import os
import tensorflow as tf


class DeepST(object):
    def __init__(self, num_channel, width, height, len_seq, external_dim, lr, model_dir,
                 len_c_conf, len_p_conf, len_t_conf, 
                 GPU_DEVICE='0'):

        self.__num_channel = num_channel
        self.__width = width
        self.__height = height
        self.__len_seq = len_seq
        self.__external_dim = external_dim
        self.__lr = lr

        self.__graph = tf.Graph()
        self.__GPU_DEVICE = GPU_DEVICE

        self.__input = {}
        self.__output = {}
        self.__op = {}
        self.__variable_init = None
        self.__saver = None
        self.__model_dir = model_dir

        # GPU Config
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.__GPU_DEVICE
        self.__config = tf.ConfigProto()
        self.__config.gpu_options.allow_growth = True

        self.__session = tf.Session(graph=self.__graph, config=self.__config)

    def build(self):
        with self.__graph.as_default():

            c_conf = tf.placeholder(tf.float32,
                                    [None, self.__height, self.__width, self.__len_seq * self.__num_channel], name='c')
            p_conf = tf.placeholder(tf.float32,
                                    [None, self.__height, self.__width, self.__len_seq * self.__num_channel], name='p')
            t_conf = tf.placeholder(tf.float32,
                                    [None, self.__height, self.__width, self.__len_seq * self.__num_channel], name='t')

            target = tf.placeholder(tf.float32, [None, self.__height, self.__width, 1], name='target')

            self.__input['c'] = c_conf.name
            self.__input['p'] = p_conf.name
            self.__input['t'] = t_conf.name
            self.__input['target'] = target.name

            # First convolution
            h_c_1 = tf.layers.conv2d(inputs=c_conf, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)
            h_p_1 = tf.layers.conv2d(inputs=p_conf, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)
            h_t_1 = tf.layers.conv2d(inputs=t_conf, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)

            # First fusion
            h_2 = tf.layers.conv2d(tf.concat([h_c_1, h_p_1, h_t_1], axis=-1),
                                   filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)

            # Stack more convolutions
            middel_output = tf.layers.conv2d(h_2, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)
            X = tf.layers.conv2d(middel_output, filters=64, kernel_size=[3, 3], padding='SAME', use_bias=True)

            # external dims
            if self.__external_dim is not None and  self.__external_dim > 0:
                external_input = tf.placeholder([None, self.__external_dim])
                self.__input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [1, 1, 1, -1]),
                                         [tf.shape(c_conf)[0], self.__height, self.__width, 1])
                X = tf.concat([X, external_dense])

            X = tf.layers.dense(X, units=1, name='prediction')

            loss = tf.sqrt(tf.reduce_mean(tf.square(X - target)), name='loss')
            train_op = tf.train.AdamOptimizer(self.__lr).minimize(loss)

            self.__output['prediction'] = X.name
            self.__output['loss'] = loss.name

            self.__op['train_op'] = train_op.name

            self.__variable_init = tf.global_variables_initializer()
            self.__saver = tf.train.Saver()

    def fit(self, C, P, T, target, external_input=None):
        feed_dict = {
            self.__graph.get_tensor_by_name(self.__input['c']): C,
            self.__graph.get_tensor_by_name(self.__input['p']): P,
            self.__graph.get_tensor_by_name(self.__input['t']): T,
            self.__graph.get_tensor_by_name(self.__input['target']): target,
        }
        if self.__external_dim is not None and self.__external_dim > 0:
            feed_dict[self.__graph.get_tensor_by_name(self.__input['external_input'])] = external_input

        l, _ = self.__session.run([self.__graph.get_tensor_by_name(self.__output['loss']),
                                   self.__graph.get_operation_by_name(self.__op['train_op'])],
                                  feed_dict=feed_dict)
        return l

    def predict(self, C, P, T, external_input=None):
        feed_dict = {
            self.__graph.get_tensor_by_name(self.__input['c']): C,
            self.__graph.get_tensor_by_name(self.__input['p']): P,
            self.__graph.get_tensor_by_name(self.__input['t']): T,
        }
        if self.__external_dim is not None and self.__external_dim > 0:
            feed_dict[self.__graph.get_tensor_by_name(self.__input['external_input'])] = external_input

        p = self.__session.run(self.__graph.get_tensor_by_name(self.__output['prediction']), feed_dict=feed_dict)
        return p

    def evaluate(self, C, P, T, target, metric, external_input=None, **kwargs):
        prediction = self.predict(C, P, T, external_input)
        return [e(prediction, target, **kwargs) for e in metric]

    def save(self, subscript):
        save_path = os.path.join(self.__model_dir, subscript)
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)
        self.__saver.save(sess=self.__session, save_path=os.path.join(save_path, subscript))

    def load(self, subscript):
        save_path = os.path.join(self.__model_dir, subscript)
        if len(os.listdir(save_path)) == 0:
            print('Model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            self.__saver.restore(self.__session, os.path.join(save_path, subscript))


# test = DeepST(1, 20, 20, 3, 0, 0, 0, '')
# test.build()