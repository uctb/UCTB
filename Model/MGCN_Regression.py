import os
import numpy as np
import tensorflow as tf
import shutil

from Model.GCLSTM_CELL import GCLSTMCell
from Model.GraphModelLayers import GAL

class MGCNRegression(object):
    def __init__(self,
                 num_node,
                 GCN_K,
                 GCN_layers,
                 num_graph,
                 external_dim,
                 T,
                 num_hidden_units,
                 num_filter_conv1x1,
                 lr,
                 code_version,
                 model_dir,
                 GPU_DEVICE='0'):

        self.__num_node = num_node
        self.__gcn_k = GCN_K
        self.__gcn_layer = GCN_layers
        self.__num_graph = num_graph
        self.__external_dim = external_dim
    
        self.__T = T
        self.__num_hidden_unit = num_hidden_units
        self.__num_filter_conv1x1 = num_filter_conv1x1
        self.__lr = lr

        self.__code_version = code_version
        self.__model_dir = model_dir
        self.__GPU_DEVICE = GPU_DEVICE

        self.__graph = tf.Graph()

        self.__input = {}
        self.__output = {}
        self.__op = {}
        self.__variable_init = None
        self.__saver = None

        self.__build = True

        # GPU Config
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.__GPU_DEVICE
        self.__config = tf.ConfigProto()
        self.__config.gpu_options.allow_growth = True

        self.__session = tf.Session(graph=self.__graph, config=self.__config)

    def build(self):
        with self.__graph.as_default():

            # Input
            input = tf.placeholder(tf.float32, [None, self.__T, None, 1], name='input')
            target = tf.placeholder(tf.float32, [None, None], name='target')
            laplace_matrix = tf.placeholder(tf.float32, [self.__num_graph, None, None], name='laplace_matrix')

            batch_size = tf.shape(input)[0]
            
            # recode input
            self.__input['input'] = input.name
            self.__input['target'] = target.name
            self.__input['laplace_matrix'] = laplace_matrix.name

            outputs_last_list = []

            for graph_index in range(self.__num_graph):
                with tf.variable_scope('gc_lstm_%s' % graph_index, reuse=False):
                    outputs_all = []
                    final_state_all = []

                    if type(self.__gcn_k) is list:
                        gc_lstm = GCLSTMCell(self.__gcn_k[graph_index], self.__gcn_layer[graph_index], self.__num_node,
                                             self.__num_hidden_unit, state_is_tuple=True,
                                             initializer=tf.contrib.layers.xavier_initializer())
                    else:
                        gc_lstm = GCLSTMCell(self.__gcn_k, self.__gcn_layer, self.__num_node,
                                             self.__num_hidden_unit, state_is_tuple=True,
                                             initializer=tf.contrib.layers.xavier_initializer())

                    gc_lstm.laplacian_matrix = tf.transpose(laplace_matrix[graph_index])

                    state = gc_lstm.zero_state(batch_size, dtype=tf.float32)

                    for i in range(0, self.__T):
                        output, state = gc_lstm(input[:, i, :, :], state)

                        outputs_all.append(output)
                        final_state_all.append(state)

                outputs_last_list.append(tf.reshape(outputs_all[-1], [-1, 1, self.__num_hidden_unit]))

            if self.__num_graph > 1:
                # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                _, gal_output_name = GAL.add_ga_layer(graph=self.__graph, inputs_name=tf.concat(outputs_last_list, axis=-2).name,
                                                      units=64, num_head=2, with_self_loop=True)
                gal_output = self.__graph.get_tensor_by_name(gal_output_name)
                pre_input = tf.reshape(tf.reduce_mean(gal_output, axis=-2), [-1, self.__num_node, 1, self.__num_hidden_unit])
            else:
                pre_input = tf.reshape(outputs_all[-1], [-1, self.__num_node, 1, self.__num_hidden_unit])

            pre_input = tf.layers.batch_normalization(pre_input, axis=-1)

            # external dims
            if self.__external_dim is not None and self.__external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self.__external_dim])
                self.__input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, tf.shape(pre_input)[1], tf.shape(pre_input)[2], 1])
                pre_input = tf.concat([pre_input, external_dense], axis=-1)

            conv1x1_output0 = tf.layers.conv2d(pre_input,
                                               filters=self.__num_filter_conv1x1,
                                               kernel_size=[1, 1],
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            pre_output = tf.layers.conv2d(conv1x1_output0,
                                          filters=1,
                                          kernel_size=[1, 1],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            prediction = tf.reshape(pre_output, [batch_size, self.__num_node], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_operation = tf.train.AdamOptimizer(self.__lr).minimize(loss_pre, name='train_op')

            # record output
            self.__output['prediction'] = prediction.name
            self.__output['loss'] = loss_pre.name

            # record train operation
            self.__op['train_op'] = train_operation.name

            self.__saver = tf.train.Saver(max_to_keep=None)
            self.__variable_init = tf.global_variables_initializer()

        self.__session.run(self.__variable_init)
        self.__build = False

    def fit(self, X, y, l_m, external_feature=None):
        if hasattr(X, 'shape') is False or hasattr(y, 'shape') is False or hasattr(l_m, 'shape') is False:
            raise ValueError('Please feed numpy array')

        if X.shape[0] != y.shape[0]:
            raise  ValueError('Expected X and y have the same batch_size, but given', X.shape, y.shape)

        feed_dict = {
            self.__graph.get_tensor_by_name(self.__input['input']): X,
            self.__graph.get_tensor_by_name(self.__input['target']): y,
            self.__graph.get_tensor_by_name(self.__input['laplace_matrix']): l_m
        }

        if self.__external_dim is not None and self.__external_dim > 0:
            feed_dict[self.__graph.get_tensor_by_name(self.__input['external_input'])] = external_feature

        l, _ = self.__session.run([self.__graph.get_tensor_by_name(self.__output['loss']),
                                   self.__graph.get_operation_by_name(self.__op['train_op'])],
                                  feed_dict=feed_dict)
        return l

    def predict(self, X, l_m, external_feature=None):
        feed_dict = {
            self.__graph.get_tensor_by_name(self.__input['input']): X,
            self.__graph.get_tensor_by_name(self.__input['laplace_matrix']): l_m
        }

        if self.__external_dim is not None and self.__external_dim > 0:
            feed_dict[self.__graph.get_tensor_by_name(self.__input['external_input'])] = external_feature

        p = self.__session.run(self.__graph.get_tensor_by_name(self.__output['prediction']),
                               feed_dict=feed_dict)
        return p

    def evaluate(self, X, y, l_m, metric, cache_volume=64, external_feature=None, threshold=0, de_normalizer=None):
        p = []
        for i in range(0, len(X), cache_volume):
            p.append(self.predict(X[i:i+cache_volume],
                                  l_m, external_feature[i:i+cache_volume]))
        p = np.concatenate(p, axis=0)
        if de_normalizer is not None:
            p = de_normalizer(p)
            y = de_normalizer(y)
        return [e(p, y, threshold=threshold) for e in metric]

    def save(self, subscript):
        save_dir = os.path.join(self.__model_dir, self.__code_version)
        save_dir_subscript = os.path.join(save_dir, subscript)
        # delete if exist
        if os.path.isdir(save_dir_subscript):
            shutil.rmtree(save_dir_subscript, ignore_errors=True)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        self.__saver.save(sess=self.__session, save_path=os.path.join(save_dir_subscript, subscript))

    def load(self, subscript):
        save_dir = os.path.join(self.__model_dir, self.__code_version)
        save_dir_subscript = os.path.join(save_dir, subscript)
        if len(os.listdir(save_dir_subscript)) == 0:
            print('Model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            self.__saver.restore(self.__session, os.path.join(save_dir_subscript, subscript))
    
    def close(self):
        self.__session.close()