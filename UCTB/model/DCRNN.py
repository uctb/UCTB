import tensorflow as tf

from ..model_unit import BaseModel
from ..model_unit import DCGRUCell

from tensorflow.contrib import legacy_seq2seq


class DCRNN(BaseModel):
    """

    References:
        - `Diffusion convolutional recurrent neural network: Data-driven traffic forecasting (Li Yaguang, et al., 2017)
          <https://arxiv.org/pdf/1707.01926.pdf>`_.
        - `A TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network (liyaguang)
          <https://github.com/liyaguang/DCRNN>`_.

    Args:
        num_node(int): Number of nodes in the graph, e.g. number of stations in NYC-Bike dataset.
        num_diffusion_matrix: Number of diffusion matrix used in model.
        num_rnn_units: Number of RNN units.
        num_rnn_layers: Number of RNN layers
        max_diffusion_step: Number of diffusion steps
        seq_len: Input sequence length
        use_curriculum_learning(bool): model's prediction (True) or the previous ground truth in training (False).
        input_dim: Dimension of the input feature
        output_dim: Dimension of the output feature
        cl_decay_steps: When use_curriculum_learning=True, cl_decay_steps is used to adjust the ratio of using ground
            true labels, where with more training steps, the ratio drops.
        target_len(int): Output sequence length.
        lr(float): Learning rate
        epsilon: epsilon in Adam
        optimizer_name(str): 'sgd' or 'Adam' optimizer
        code_version(str): Current version of this model code, which will be used as filename for saving the model
        model_dir(str): The directory to store model files. Default:'model_dir'.
        gpu_device(str): To specify the GPU to use. Default: '0'.
    """
    def __init__(self,
                 num_node,
                 num_diffusion_matrix,
                 num_rnn_units=64,
                 num_rnn_layers=1,
                 max_diffusion_step=2,
                 seq_len=6,
                 use_curriculum_learning=False,
                 input_dim=1,
                 output_dim=1,
                 cl_decay_steps=1000,
                 target_len=1,
                 lr=1e-4,
                 epsilon=1e-3,
                 optimizer_name='Adam',
                 code_version='DCRNN-QuickStart',
                 model_dir='model_dir',
                 gpu_device='0', **kwargs):

        super(DCRNN, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)

        self._num_node = num_node
        self._num_diffusion_matrix = num_diffusion_matrix
        self._num_rnn_units = num_rnn_units
        self._num_rnn_layers = num_rnn_layers
        self._max_diffusion_step = max_diffusion_step
        self._seq_len = seq_len
        self._use_curriculum_learning = use_curriculum_learning
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._target_len = target_len
        self._cl_decay_steps = cl_decay_steps
        self._optimizer_name = optimizer_name
        self._lr = lr
        # self._batch_size = batch_size
        self._epsilon = epsilon

    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():
            inputs = tf.placeholder(tf.float32, shape=(None, self._seq_len,
                                                       self._num_node, self._input_dim), name='inputs')
            labels = tf.placeholder(tf.float32, shape=(None, self._target_len,
                                                       self._num_node, self._output_dim), name='labels')

            diffusion_matrix = tf.placeholder(tf.float32, shape=(self._num_diffusion_matrix, self._num_node,
                                                                 self._num_node), name='diffusion_matrix')

            batch_size = tf.shape(inputs)[0]

            self._input['inputs'] = inputs.name
            self._input['target'] = labels.name
            self._input['diffusion_matrix'] = diffusion_matrix.name

            go_symbol = tf.zeros(shape=(tf.shape(inputs)[0], self._num_node * self._output_dim))

            cell = DCGRUCell(self._num_rnn_units, self._input_dim, self._num_diffusion_matrix, diffusion_matrix,
                             max_diffusion_step=self._max_diffusion_step, num_node=self._num_node)

            cell_with_projection = DCGRUCell(self._num_rnn_units, self._input_dim,
                                             self._num_diffusion_matrix, diffusion_matrix,
                                             max_diffusion_step=self._max_diffusion_step,
                                             num_node=self._num_node, num_proj=self._output_dim)

            encoding_cells = [cell] * self._num_rnn_layers
            decoding_cells = [cell] * (self._num_rnn_layers - 1) + [cell_with_projection]
            encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
            decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

            global_step = tf.train.get_or_create_global_step()

            # Outputs: (batch_size, timesteps, num_node, output_dim)
            with tf.variable_scope('DCRNN_SEQ'):
                inputs_unstack = tf.unstack(tf.reshape(inputs, (batch_size,
                                                                self._seq_len, self._num_node * self._input_dim)),
                                            axis=1)
                labels_unstack = tf.unstack(
                    tf.reshape(labels[..., :self._output_dim],
                               (batch_size, self._target_len, self._num_node * self._output_dim)), axis=1)
                labels_unstack.insert(0, go_symbol)

                def _compute_sampling_threshold(global_step, k):
                    """
                    Computes the sampling probability for scheduled sampling using inverse sigmoid.
                    global_step:
                    k:
                    :return:
                    """
                    return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

                def _loop_function_train(prev, i):
                    # Return either the model's prediction or the previous ground truth in training.
                    if self._use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = _compute_sampling_threshold(global_step, self._cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels_unstack[i], lambda: prev)
                    else:
                        result = labels_unstack[i]
                    return result

                def _loop_function_test(prev, i):
                    # Return the prediction of the model in testing.
                    return prev

                a, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs_unstack, dtype=tf.float32)

                with tf.variable_scope('train', reuse=False):
                    train_outputs, _ = legacy_seq2seq.rnn_decoder(labels_unstack, enc_state, decoding_cells,
                                                                  loop_function=_loop_function_train)
                with tf.variable_scope('text', reuse=True):
                    test_outputs, _ = legacy_seq2seq.rnn_decoder(labels_unstack, enc_state, decoding_cells,
                                                                 loop_function=_loop_function_test)

            # Project the output to output_dim.
            train_outputs = tf.stack(train_outputs[:-1], axis=1)
            test_outputs = tf.stack(test_outputs[:-1], axis=1)

            # Configure optimizer
            optimizer = tf.train.AdamOptimizer(self._lr, epsilon=float(self._epsilon))

            if self._optimizer_name == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self._lr)

            loss = tf.sqrt(tf.reduce_mean(tf.square(train_outputs - labels[:, :, :, 0])))

            train_op = optimizer.minimize(loss)

            self._output['prediction'] = test_outputs.name
            self._output['loss'] = loss.name
            self._op['train_op'] = train_op.name

        super(DCRNN, self).build(init_vars=init_vars, max_to_keep=5)

    # Define your '_get_feed_dict function‘, map your input to the tf-model
    def _get_feed_dict(self,
                       inputs,
                       diffusion_matrix,
                       target=None,):
        feed_dict = {
            'inputs': inputs,
            'diffusion_matrix': diffusion_matrix,
        }
        if target is not None:
            feed_dict['target'] = target
        return feed_dict
