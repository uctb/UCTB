import os
import shutil
import tensorflow as tf


class ST_ResNet(object):
    def __init__(self, num_channel, width, height, external_dim, num_residual_unit, lr, model_dir,
                 len_c_conf, len_p_conf, len_t_conf, code_version,
                 conv_filters=64,
                 GPU_DEVICE='0'):

        self.__num_channel = num_channel
        self.__width = width
        self.__height = height
        self.__len_c_conf = len_c_conf
        self.__len_p_conf = len_p_conf
        self.__len_t_conf = len_t_conf
        self.__conv_filters = conv_filters
        self.__external_dim = external_dim
        self.__num_residual_unit = num_residual_unit
        self.__lr = lr

        self.__graph = tf.Graph()
        self.__GPU_DEVICE = GPU_DEVICE

        self.__input = {}
        self.__output = {}
        self.__op = {}
        self.__variable_init = None
        self.__saver = None
        self.__model_dir = model_dir
        self.__code_version = code_version
        self.__save_dir = os.path.join(self.__model_dir, self.__code_version)

        # GPU Config
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.__GPU_DEVICE
        self.__config = tf.ConfigProto()
        self.__config.gpu_options.allow_growth = True

        self.__session = tf.Session(graph=self.__graph, config=self.__config)

    def build(self):
        with self.__graph.as_default():

            target_conf = []

            if self.__len_c_conf is not None and self.__len_c_conf > 0:
                c_conf = tf.placeholder(tf.float32, [None, self.__height, self.__width, self.__len_c_conf * self.__num_channel], name='c')
                self.__input['c'] = c_conf.name
                target_conf.append(c_conf)

            if self.__len_p_conf is not None and self.__len_p_conf > 0:
                p_conf = tf.placeholder(tf.float32, [None, self.__height, self.__width, self.__len_p_conf * self.__num_channel], name='p')
                self.__input['p'] = p_conf.name
                target_conf.append(p_conf)

            if self.__len_t_conf is not None and self.__len_t_conf > 0:
                t_conf = tf.placeholder(tf.float32, [None, self.__height, self.__width, self.__len_t_conf * self.__num_channel], name='t')
                self.__input['t'] = t_conf.name
                target_conf.append(t_conf)
            
            target = tf.placeholder(tf.float32, [None, self.__height, self.__width, 1], name='target')

            self.__input['target'] = target.name

            outputs = []
            for conf in target_conf:

                residual_input = tf.layers.conv2d(conf, filters=self.__conv_filters, kernel_size=[3, 3],
                                                  padding='SAME', activation=tf.nn.relu)

                def residual_unit(x):
                    residual_output = tf.nn.relu(x)
                    residual_output = tf.layers.conv2d(residual_output, filters=self.__conv_filters, kernel_size=[3, 3], padding='SAME')
                    residual_output = tf.nn.relu(residual_output)
                    residual_output = tf.layers.conv2d(residual_output, filters=self.__conv_filters, kernel_size=[3, 3], padding='SAME')
                    return residual_output + x

                for i in range(self.__num_residual_unit):
                    residual_input = residual_unit(residual_input)

                outputs.append(tf.layers.conv2d(tf.nn.relu(residual_input), filters=self.__conv_filters, kernel_size=[3, 3], padding='SAME'))

            if len(outputs) == 1:
                X = outputs[0]
            else:
                fusion_weight = tf.Variable(tf.random_normal([len(outputs), ]))
                for i in range(len(outputs)):
                    outputs[i] = fusion_weight[i] * outputs[i]
                X = tf.reduce_sum(outputs, axis=0)

            # external dims
            if self.__external_dim is not None and  self.__external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self.__external_dim])
                self.__input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, self.__height, self.__width, 1])
                X = tf.concat([X, external_dense], axis=-1)

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
        save_dir_subscript = os.path.join(self.__save_dir, subscript)
        # delete if exist
        if os.path.isdir(save_dir_subscript):
            shutil.rmtree(save_dir_subscript, ignore_errors=True)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        self.__saver.save(sess=self.__session, save_path=os.path.join(save_dir_subscript, subscript))

    def load(self, subscript):
        save_dir_subscript = os.path.join(self.__save_dir, subscript)
        if len(os.listdir(save_dir_subscript)) == 0:
            print('Model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            self.__saver.restore(self.__session, save_dir_subscript)

# num_channel, width, height, external_dim, num_residual_unit, lr, model_dir,
                 # len_c_conf, len_p_conf, len_t_conf,
# test = ST_ResNet(1, 20, 20, 3, 1, 0.0001, '', 3, 3, 3)
# test.build()