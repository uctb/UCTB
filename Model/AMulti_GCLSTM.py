import numpy as np
import tensorflow as tf

from ModelUnit.GCLSTM_CELL import GCLSTMCell
from ModelUnit.GraphModelLayers import GAL

from Model.BaseModel import BaseModel


class AMulti_GCLSTM(BaseModel):
    def __init__(self,
                 num_node,
                 GCN_K,
                 GCN_layers,
                 GCLSTM_layers,
                 num_graph,
                 external_dim,
                 T,
                 gal_units,
                 gal_num_heads,
                 num_hidden_units,
                 num_filter_conv1x1,
                 lr,
                 code_version,
                 model_dir,
                 GPU_DEVICE='0'):

        super(AMulti_GCLSTM, self).__init__(code_version=code_version, model_dir=model_dir, GPU_DEVICE=GPU_DEVICE)

        self._num_node = num_node
        self._gcn_k = GCN_K
        self._gcn_layer = GCN_layers
        self._gal_units = gal_units
        self._gal_num_heads = gal_num_heads
        self.__gclstm_layers = GCLSTM_layers
        self._num_graph = num_graph
        self._external_dim = external_dim

        self._T = T
        self._num_hidden_unit = num_hidden_units
        self._num_filter_conv1x1 = num_filter_conv1x1
        self._lr = lr

    def build(self):
        with self._graph.as_default():

            # Input
            input = tf.placeholder(tf.float32, [None, self._T, None, 1], name='input')
            target = tf.placeholder(tf.float32, [None, None], name='target')
            laplace_matrix = tf.placeholder(tf.float32, [self._num_graph, None, None], name='laplace_matrix')

            batch_size = tf.shape(input)[0]

            # recode input
            self._input['input'] = input.name
            self._input['target'] = target.name
            self._input['laplace_matrix'] = laplace_matrix.name

            outputs_last_list = []

            for graph_index in range(self._num_graph):
                with tf.variable_scope('gc_lstm_%s' % graph_index, reuse=False):
                    outputs_all = []

                    if type(self._gcn_k) is list:
                        gc_lstm_cells = [GCLSTMCell(self._gcn_k[graph_index], self._gcn_layer[graph_index], self._num_node,
                                                    self._num_hidden_unit, state_is_tuple=True,
                                                    initializer=tf.contrib.layers.xavier_initializer())
                                         for _ in range(self.__gclstm_layers)]
                    else:
                        gc_lstm_cells = [
                            GCLSTMCell(self._gcn_k, self._gcn_layer, self._num_node,
                                       self._num_hidden_unit, state_is_tuple=True,
                                       initializer=tf.contrib.layers.xavier_initializer())
                            for _ in range(self.__gclstm_layers)]

                    for cell in gc_lstm_cells:
                        cell.laplacian_matrix = tf.transpose(laplace_matrix[graph_index])

                    cell_state_list = [cell.zero_state(batch_size, dtype=tf.float32) for cell in gc_lstm_cells]

                    for i in range(0, self._T):

                        output = input[:, i, :, :]

                        for cell_index in range(len(gc_lstm_cells)):

                            output, cell_state_list[cell_index] = gc_lstm_cells[cell_index](output, cell_state_list[cell_index])

                        outputs_all.append(output)

                outputs_last_list.append(tf.reshape(outputs_all[-1], [-1, 1, self._num_hidden_unit]))

            if self._num_graph > 1:
                # (graph, inputs_name, units, num_head, activation=tf.nn.leaky_relu)
                _, gal_output_name = GAL.add_ga_layer(graph=self._graph,
                                                      inputs_name=tf.concat(outputs_last_list, axis=-2).name,
                                                      units=self._gal_units, num_head=self._gal_num_heads,
                                                      with_self_loop=True)
                gal_output = self._graph.get_tensor_by_name(gal_output_name)
                pre_input = tf.reshape(tf.reduce_mean(gal_output, axis=-2),
                                       [-1, self._num_node, 1, self._num_hidden_unit])
            else:
                pre_input = tf.reshape(outputs_last_list[-1], [-1, self._num_node, 1, self._num_hidden_unit])

            pre_input = tf.layers.batch_normalization(pre_input, axis=-1)

            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_input'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, tf.shape(pre_input)[1], tf.shape(pre_input)[2], 1])
                pre_input = tf.concat([pre_input, external_dense], axis=-1)

            conv1x1_output0 = tf.layers.conv2d(pre_input,
                                               filters=self._num_filter_conv1x1,
                                               kernel_size=[1, 1],
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            pre_output = tf.layers.conv2d(conv1x1_output0,
                                          filters=1,
                                          kernel_size=[1, 1],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            prediction = tf.reshape(pre_output, [batch_size, self._num_node], name='prediction')

            loss_pre = tf.sqrt(tf.reduce_mean(tf.square(target - prediction)), name='loss')
            train_operation = tf.train.AdamOptimizer(self._lr).minimize(loss_pre, name='train_op')

            # record output
            self._output['prediction'] = prediction.name
            self._output['loss'] = loss_pre.name

            # record train operation
            self._op['train_op'] = train_operation.name

            self._saver = tf.train.Saver(max_to_keep=None)
            self._variable_init = tf.global_variables_initializer()

            # Add summary
            self._summary = self._summary_histogram().name

        self._session.run(self._variable_init)
        self._build = False


if __name__ == '__main__':

    MGCNRegression_Obj = AMulti_GCLSTM(num_node=500, GCN_K=1, GCN_layers=1, GCLSTM_layers=1,
                                        num_graph=1,
                                        external_dim=30,
                                        gal_units=64, gal_num_heads=2,
                                        T=6, num_filter_conv1x1=32, num_hidden_units=64,
                                        lr=1e-4, code_version='Debug', GPU_DEVICE='0', model_dir='')
    MGCNRegression_Obj.build()