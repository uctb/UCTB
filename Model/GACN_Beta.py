import numpy as np
import tensorflow as tf

from ModelUnit.GraphModelLayers import GAL, GCL
from Model.BaseModel import BaseModel


class GACN(BaseModel):
    def __init__(self,
                 num_node,
                 gcl_k,
                 gcl_layers,
                 gal_num_heads,
                 gal_layers,
                 gal_units,
                 dense_units,
                 time_embedding_dim,
                 input_dim,
                 T,
                 lr,
                 code_version,
                 model_dir,
                 GPU_DEVICE='0'):

        super(GACN, self).__init__(code_version=code_version, model_dir=model_dir, GPU_DEVICE=GPU_DEVICE)

        self._num_node = num_node

        self._input_dim = input_dim

        self._gal_num_heads = gal_num_heads
        self._gal_layers = gal_layers
        self._gal_units = gal_units
        self._time_embedding_dim = time_embedding_dim

        self._gcl_k = gcl_k
        self._gcl_layers = gcl_layers

        self._dense_units = dense_units

        self._T = T
        self._lr = lr

    def build(self):
        with self._graph.as_default():
            # Input
            input_raw = tf.placeholder(tf.float32, [None, self._T, None, self._input_dim], name='input_hour')

            input = tf.transpose(input_raw, perm=[0, 2, 1, 3])

            target = tf.placeholder(tf.float32, [None, None, 1], name='target')
            laplace_matrix = tf.placeholder(tf.float32, [None, None], name='laplace_matrix')

            # recode input
            self._input['input'] = input_raw.name
            self._input['target'] = target.name
            self._input['laplace_matrix'] = laplace_matrix.name

            if self._time_embedding_dim and self._time_embedding_dim > 0:

                time_embedding = tf.placeholder(tf.float32, [self._T, self._time_embedding_dim], name='time_embedding')

                # recode input
                self._input['time_embedding'] = time_embedding.name

                time_embedding = tf.reshape(time_embedding, [1, 1, self._T, self._time_embedding_dim])
                time_embedding = tf.tile(time_embedding,
                                         [tf.shape(input)[0], tf.shape(input)[1], 1, 1])

                input = tf.concat((input, time_embedding), axis=-1)

                attention_input = tf.reshape(input, [-1, self._T, self._input_dim + self._time_embedding_dim])

            else:

                attention_input = tf.reshape(input, [-1, self._T, self._input_dim])

            attention_output_list = []
            for loop_index in range(self._gal_layers):
                with tf.variable_scope('res_gal_%s' % loop_index, reuse=False):
                    attention_output_name = GAL.add_residual_ga_layer(self._graph,
                                                                      attention_input.name,
                                                                      num_head=self._gal_num_heads,
                                                                      units=self._gal_units)
                    attention_input = self._graph.get_tensor_by_name(attention_output_name)
                    attention_output_list.append(attention_input)

            attention_output = tf.reshape(attention_output_list[-1],
                                          [tf.shape(input)[0], tf.shape(input)[1],
                                           self._T, attention_input.get_shape()[-1]])

            # GCN
            gcn_input_feature = tf.reduce_mean(attention_output, axis=-2)

            gcn_output_name = GCL.add_gc_layer(self._graph, gcn_input_feature.name, self._gcl_k, laplace_matrix)

            gcn_output = self._graph.get_tensor_by_name(gcn_output_name)

            middle_output = tf.keras.layers.Dense(units=self._dense_units)(gcn_output)
            prediction = tf.keras.layers.Dense(units=1)(middle_output)

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_operation = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_operation.name

            self._saver = tf.train.Saver(max_to_keep=None)
            self._variable_init = tf.global_variables_initializer()

            self.trainable_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

            print('Trainable Variables', self.trainable_vars)

            # Add summary
            self._summary = self._summary_histogram().name

        self._session.run(self._variable_init)