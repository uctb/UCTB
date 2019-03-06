import os
import shutil
from Model.BaseModel import BaseModel
import tensorflow as tf


class ST_ResNet(BaseModel):
    def __init__(self, num_channel, width, height, external_dim, num_residual_unit, lr, model_dir,
                 len_c_conf, len_p_conf, len_t_conf, code_version,
                 conv_filters=64,
                 GPU_DEVICE='0'):

        super(ST_ResNet, self).__init__(code_version=code_version, model_dir=model_dir, GPU_DEVICE=GPU_DEVICE)

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

    def build(self):
        with self._graph.as_default():

            target_conf = []

            if self.__len_c_conf is not None and self.__len_c_conf > 0:
                c_conf = tf.placeholder(tf.float32, [None, self.__height, self.__width, self.__len_c_conf * self.__num_channel], name='c')
                self._input['c'] = c_conf.name
                target_conf.append(c_conf)

            if self.__len_p_conf is not None and self.__len_p_conf > 0:
                p_conf = tf.placeholder(tf.float32, [None, self.__height, self.__width, self.__len_p_conf * self.__num_channel], name='p')
                self._input['p'] = p_conf.name
                target_conf.append(p_conf)

            if self.__len_t_conf is not None and self.__len_t_conf > 0:
                t_conf = tf.placeholder(tf.float32, [None, self.__height, self.__width, self.__len_t_conf * self.__num_channel], name='t')
                self._input['t'] = t_conf.name
                target_conf.append(t_conf)
            
            target = tf.placeholder(tf.float32, [None, self.__height, self.__width, 1], name='target')

            self._input['target'] = target.name

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
                self._input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, self.__height, self.__width, 1])
                X = tf.concat([X, external_dense], axis=-1)

            X = tf.layers.dense(X, units=1, name='prediction')
            
            loss = tf.sqrt(tf.reduce_mean(tf.square(X - target)), name='loss')
            train_op = tf.train.AdamOptimizer(self.__lr).minimize(loss)

            self._output['prediction'] = X.name
            self._output['loss'] = loss.name

            self._op['train_op'] = train_op.name

            self._variable_init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        self._session.run(self._variable_init)
        self._build = False