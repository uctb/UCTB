import tensorflow as tf
from tensorflow.contrib.framework import nest
from ..model_unit import BaseModel


class GeoMAN(BaseModel):
    """Multi-level Attention Networks for Geo-sensory Time Series Prediction (GeoMAN)

            GeoMAN consists of two major parts: 1) A multi-level attention mechanism (including both local and global
            spatial attentions in encoder and temporal attention in decoder) to model the dynamic spatio-temporal
            dependencies; 2) A general fusion module to incorporate the external factors from different domains (e.g.,
            meteorology, time of day and land use).

            References:
                - `GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction (Liang Yuxuan, et al., 2018)
                  <https://www.ijcai.org/proceedings/2018/0476.pdf>`_.
                - `An easy implement of GeoMAN using TensorFlow (yoshall & CastleLiang)
                  <https://github.com/yoshall/GeoMAN>`_.

            Args:
                total_sensers (int): The number of total sensors used in global attention mechanism.
                input_dim (int): The number of dimensions of the target sensor's input.
                external_dim (int): The number of dimensions of the external features.
                output_dim (int): The number of dimensions of the target sensor's output.
                input_steps (int): The length of historical input data, a.k.a, input timesteps.
                output_steps (int): The number of steps that need prediction by one piece of history data, a.k.a, output
                    timesteps. Have to be 1 now.
                n_stacked_layers (int): The number of LSTM layers stacked in both encoder and decoder (These two are the
                    same). Default: 2
                n_encoder_hidden_units (int): The number of hidden units in each layer of encoder. Default: 128
                n_decoder_hidden_units (int): The number of hidden units in each layer of decoder. Default: 128
                dropout_rate (float): Dropout rate of LSTM layers in both encoder and decoder. Default: 0.3
                lr (float): Learning rate. Default: 0.001
                gc_rate (float): A clipping ratio for all the gradients. This operation normalizes all gradients so that
                    their L2-norms are less than or equal to ``gc_rate``. Default: 2.5
                code_version (str): Current version of this model code. Default: 'GeoMAN-QuickStart'
                model_dir (str): The directory to store model files. Default:'model_dir'
                gpu_device (str): To specify the GPU to use. Default: '0'
                **kwargs (dict): Reserved for future use. May be used to pass parameters to class ``BaseModel``.
            """
    def __init__(self,
                 total_sensers,
                 input_dim,
                 external_dim,
                 output_dim,
                 input_steps,
                 output_steps,
                 n_stacked_layers=2,
                 n_encoder_hidden_units=128,
                 n_decoder_hidden_units=128,
                 dropout_rate=0.3,
                 lr=0.001,
                 gc_rate=2.5,
                 code_version='GeoMAN-QuickStart',
                 model_dir='model_dir',
                 gpu_device='0',
                 **kwargs):

        super(GeoMAN, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)

        # Architecture
        self._n_stacked_layers = n_stacked_layers
        self._n_encoder_hidden_units = n_encoder_hidden_units
        self._n_decoder_hidden_units = n_decoder_hidden_units
        self._n_output_decoder = output_dim  # n_output_decoder

        self._n_steps_encoder = input_steps  # encoder_steps
        self._n_steps_decoder = output_steps  # decoder_steps
        self._n_input_encoder = input_dim  # n_input_encoder
        self._n_sensers = total_sensers  # n_sensers
        self._n_external_input = external_dim  # external_dim

        # Hyperparameters
        self._dropout_rate = dropout_rate
        self._lr = lr
        self._gc_rate = gc_rate

    def build(self, init_vars=True, max_to_keep=5):
        with self._graph.as_default():
            with tf.variable_scope('inputs'):
                local_features = tf.placeholder(tf.float32, shape=[None, self._n_steps_encoder, self._n_input_encoder],
                                                name='local_features')
                global_features = tf.placeholder(tf.float32, shape=[None, self._n_steps_encoder, self._n_sensers],
                                                 name='global_features')
                external_features = tf.placeholder(tf.float32,
                                                   shape=[None, self._n_steps_decoder, self._n_external_input],
                                                   name='external_features')
                local_attn_states = tf.placeholder(tf.float32,
                                                   shape=[None, self._n_input_encoder, self._n_steps_encoder],
                                                   name='local_attn_states')
                global_attn_states = tf.placeholder(tf.float32, shape=[None, self._n_sensers, self._n_input_encoder,
                                                                       self._n_steps_encoder],
                                                    name='global_attn_states')
            with tf.variable_scope('ground_truth'):
                targets = tf.placeholder(tf.float32, [None, self._n_steps_decoder, self._n_output_decoder])

            self._input['local_features'] = local_features.name
            self._input['global_features'] = global_features.name
            self._input['external_features'] = external_features.name
            self._input['local_attn_states'] = local_attn_states.name
            self._input['global_attn_states'] = global_attn_states.name
            self._input['targets'] = targets.name

            predict_layer = tf.keras.layers.Dense(units=self._n_output_decoder,
                                                  kernel_initializer=tf.truncated_normal_initializer,
                                                  bias_initializer=tf.constant_initializer(0.),
                                                  use_bias=True)

            def _build_cells(n_hidden_units):
                cells = []
                for i in range(self._n_stacked_layers):
                    with tf.variable_scope(f'LSTM_{i}'):
                        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,
                                                            forget_bias=1.0,
                                                            state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0 - self._dropout_rate)
                        cells.append(cell)
                encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
                return encoder_cell

            def _loop_function(prev):
                """loop function used in the decoder to generate the next input"""
                return predict_layer(prev)

            def _get_MSE_loss(y_true, y_pred):
                return tf.reduce_mean(tf.pow(y_true - y_pred, 2), name='MSE_loss')

            def _get_l2reg_loss():
                # l2 loss
                reg_loss = 0
                for tf_var in tf.trainable_variables():
                    if 'kernel:' in tf_var.name or 'bias:' in tf_var.name:
                        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                return 0.001 * reg_loss

            def _spatial_attention(local_features,  # x and X
                                   global_features,
                                   local_attention_states,
                                   global_attention_states,
                                   encoder_cells,  # to acquire h_{t-1}, s_{t-1}
                                   ):
                batch_size = tf.shape(local_features[0])[0]
                output_size = encoder_cells.output_size
                with tf.variable_scope('spatial_attention'):
                    with tf.variable_scope('local_spatial_attn'):
                        local_attn_length = local_attention_states.get_shape()[1].value  # n_input_encoder
                        local_attn_size = local_attention_states.get_shape()[2].value  # n_steps_encoder
                        local_attn = tf.zeros([batch_size, local_attn_length])

                        #  Add local features in attention
                        x_ik = tf.reshape(local_attention_states,
                                          [-1, local_attn_length, 1, local_attn_size])  # features
                        Ul = tf.get_variable('spati_atten_Ul', [1, 1, local_attn_size, local_attn_size])
                        Ul_x = tf.nn.conv2d(x_ik, Ul, [1, 1, 1, 1], 'SAME')  # U_l * x^{i,k}
                        vl = tf.get_variable('spati_atten_vl', [local_attn_size])  # v_l

                        def _local_spatial_attention(query):
                            # If the query is a tuple (when stacked RNN/LSTM), flatten it
                            if hasattr(query, "__iter__"):
                                query_list = nest.flatten(query)
                                for q in query_list:
                                    ndims = q.get_shape().ndims
                                    if ndims:
                                        assert ndims == 2
                                query = tf.concat(query_list, 1)
                            with tf.variable_scope('local_spatial_attn_Wl'):
                                h_s = query
                                Wl_hs_bl = tf.keras.layers.Dense(units=local_attn_size, use_bias=True)(h_s)
                                Wl_hs_bl = tf.reshape(Wl_hs_bl, [-1, 1, 1, local_attn_size])
                                score = tf.reduce_sum(vl * tf.nn.tanh(Wl_hs_bl + Ul_x),
                                                      [2, 3])  # ! Ux is a 4 dims matrix, have to use reduce_sum here
                                attention_weights = tf.nn.softmax(score)
                            return attention_weights

                    with tf.variable_scope('global_spatial_attn'):
                        global_attn_length = global_attention_states.get_shape()[1].value  # n_sensor
                        global_n_input = global_attention_states.get_shape()[2].value  # n_input_dim
                        global_attn_size = global_attention_states.get_shape()[3].value  # n_input_dim
                        global_attn = tf.zeros([batch_size, global_attn_length])

                        # Add global features in attention
                        Xl = tf.reshape(global_attention_states,
                                        [-1, global_attn_length, global_n_input, global_attn_size])
                        Wg_ug = tf.get_variable('spati_atten_Wg_ug',
                                                [1, global_n_input, global_attn_size, global_attn_size])
                        Wg_Xl_ug = tf.nn.conv2d(Xl, Wg_ug, [1, 1, 1, 1], 'SAME')
                        vg = tf.get_variable('spati_atten_vg', [local_attn_size])

                        # TODO: add U_g * y^l here, where y^l is the first column of local inputs.

                        def _global_spatial_attention(query):
                            if hasattr(query, "__iter__"):
                                query_list = nest.flatten(query)
                                for q in query_list:  # Check that ndims == 2 if specified.
                                    ndims = q.get_shape().ndims
                                    if ndims:
                                        assert ndims == 2
                                query = tf.concat(query_list, 1)
                            with tf.variable_scope('global_spatial_attn_Wl'):
                                h_s = query
                                Wg_hs_bg = tf.keras.layers.Dense(units=global_attn_size, use_bias=True)(h_s)
                                Wg_hs_bg = tf.reshape(Wg_hs_bg, [-1, 1, 1, global_attn_size])
                                score = tf.reduce_sum(vg * tf.nn.tanh(Wg_hs_bg + Wg_Xl_ug), [2, 3])
                                attention_weights = tf.nn.softmax(score)
                                # Sometimes it's not easy to find a measurement to denote similarity between sensors,
                                # here we omit such prior knowledge in eq.[4].
                                # You can use "a = nn_ops.softmax((1-lambda)*s + lambda*sim)" to encode similarity info,
                                # where:
                                #     sim: a vector with length n_sensors, describing the sim between the target sensor and the others
                                #     lambda: a trade-off.
                                # attention_weights = tf.softmax((1-self.sm_rate)*score+self.sm_rate*self.similarity_graph)
                            return attention_weights

                    # Init
                    zeros = [tf.zeros([batch_size, output_size]) for i in range(2)]
                    initial_state = [zeros for _ in range(len(encoder_cells._cells))]
                    state = initial_state

                    # For each timestep
                    outputs = []
                    attn_weights = []
                    for i, (local_input, global_input) in enumerate(zip(local_features, global_features)):
                        if i > 0: tf.get_variable_scope().reuse_variables()

                        local_context_vector = local_attn * local_input
                        global_context_vector = global_attn * global_input
                        x_t = tf.concat([local_context_vector, global_context_vector], axis=1)
                        encoder_output, state = encoder_cells(x_t, state)  # Update states

                        with tf.variable_scope('local_spatial_attn'):
                            local_attn = _local_spatial_attention(state)
                        with tf.variable_scope('global_spatial_attn'):
                            global_attn = _global_spatial_attention(state)
                        attn_weights.append((local_attn, global_attn))
                        outputs.append(encoder_output)

                return outputs, state, attn_weights

            def _temporal_attention(decoder_inputs,
                                    external_features,
                                    inital_states,  # the first time, the output of encoder
                                    attention_states,  # h_o
                                    decoder_cells):
                batch_size = tf.shape(decoder_inputs[0])[0]
                output_size = decoder_cells.output_size
                input_size = decoder_inputs[0].get_shape().with_rank(2)[1]  # ?
                state = inital_states
                with tf.variable_scope('temperal_attention'):
                    attn_length = attention_states.get_shape()[1].value
                    attn_size = attention_states.get_shape()[2].value

                    h_o = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
                    W_d = tf.get_variable('temperal_attn_Wd', [1, 1, attn_size, attn_size])
                    W_h = tf.nn.conv2d(h_o, W_d, [1, 1, 1, 1], 'SAME')
                    v_d = tf.get_variable('temperal_attn_vd', [attn_size])

                    def _attention(query):
                        if hasattr(query, "__iter__"):
                            query_list = nest.flatten(query)
                            for q in query_list:  # Check that ndims == 2 if specified.
                                ndims = q.get_shape().ndims
                                if ndims:
                                    assert ndims == 2
                            query = tf.concat(query_list, 1)
                        with tf.variable_scope('attention'):
                            d_s = query
                            W_ds_b = tf.keras.layers.Dense(units=attn_size, use_bias=True)(d_s)
                            W_ds_b = tf.reshape(W_ds_b, [-1, 1, 1, attn_size])
                            score = tf.reduce_sum(v_d * tf.nn.tanh(W_ds_b + W_h), [2, 3])
                            attention_weights = tf.nn.softmax(score)
                            context_vector = tf.reduce_sum(
                                tf.reshape(attention_weights, [-1, attn_length, 1, 1]) * h_o, [1, 2])
                            context_vector = tf.reshape(context_vector, [-1, attn_size])
                        return context_vector

                    # Init
                    inital_attn = tf.zeros([batch_size, output_size])
                    attn = inital_attn
                    outputs = []

                    prev_decoder_output = None  # d_{t-1}
                    for i, (decoder_input, external_input) in enumerate(zip(decoder_inputs, external_features)):
                        if i > 0: tf.get_variable_scope().reuse_variables()
                        if prev_decoder_output is not None and _loop_function is not None:
                            with tf.variable_scope('loop_function', reuse=True):
                                decoder_input = _loop_function(prev_decoder_output)
                        x = tf.concat([decoder_input, external_input, attn], axis=1)
                        x = tf.keras.layers.Dense(units=input_size, use_bias=True)(x)
                        decoder_output, state = decoder_cells(x, state)
                        # Update attention weights
                        attn = _attention(state)
                        # Attention output projection
                        with tf.variable_scope("attn_output_projection"):
                            x = tf.concat([decoder_output, attn], axis=1)
                            output = tf.keras.layers.Dense(units=output_size, use_bias=True)(x)
                        outputs.append(output)
                        prev_decoder_output = output
                return outputs, state

            # Handle data
            local_features, global_features, external_features, targets, decoder_inputs = input_transform(
                local_features, global_features, external_features, targets)

            with tf.variable_scope('GeoMAN'):
                with tf.variable_scope('encoder'):
                    encoder_cells = _build_cells(self._n_encoder_hidden_units)
                    encoder_outputs, encoder_state, attn_weights = _spatial_attention(local_features,
                                                                                      global_features,
                                                                                      local_attn_states,
                                                                                      global_attn_states,
                                                                                      encoder_cells)
                    top_states = [tf.reshape(e, [-1, 1, encoder_cells.output_size]) for e in encoder_outputs]
                    attention_states = tf.concat(top_states, 1)

                with tf.variable_scope('decoder'):
                    decoder_cells = _build_cells(self._n_decoder_hidden_units)
                    decoder_outputs, states = _temporal_attention(decoder_inputs,
                                                                  external_features,
                                                                  encoder_state,
                                                                  attention_states,
                                                                  decoder_cells)

                with tf.variable_scope('prediction'):
                    predictions = []
                    for decoder_output in decoder_outputs:
                        predictions.append(predict_layer(decoder_output))
                    predictions = tf.stack(predictions, axis=1, name='predictions')

                with tf.variable_scope('loss'):
                    targets = tf.stack(targets, axis=1, name='targets')
                    loss = tf.add(_get_MSE_loss(targets, predictions), _get_l2reg_loss(), name='loss')

                with tf.variable_scope('train_op'):
                    global_step = tf.train.get_or_create_global_step()
                    optimizer = tf.train.AdamOptimizer(self._lr)
                    gradients, variables = zip(*optimizer.compute_gradients(loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, self._gc_rate)  # clip norm
                    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

                # record output
                self._output['prediction'] = predictions.name
                self._output['loss'] = loss.name
                # record op
                self._op['train_op'] = train_op.name

        super(GeoMAN, self).build(init_vars=init_vars, max_to_keep=5)

    def _get_feed_dict(self,
                       local_features,
                       global_features,
                       local_attn_states,
                       global_attn_states,
                       external_features,
                       targets):
        """The method to get feet dict for tensorflow model.

        Users may modify this method according to the format of input.

        Args:
            local_features (np.ndarray): All the time series generated by the target sensor i, including one target
                series and other feature series, with shape `(batch, input_steps, input_dim)`.
            global_features (np.ndarray): Target series generated by all the sensors, with shape `(batch, input_steps,
                total_sensors)`.
            local_attn_states (np.ndarray): Equals to ``local_features`` swapped ``input_steps`` and ``input_dim`` axis,
                with shape `(batch, input_dim, input_steps)`.
            global_attn_states (np.ndarray): All time series generated by all sensors, with shape `(batch,
                total_sensors, input_dim, input_steps)`.
            external_features (np.ndarray): Fused external factors, e.g., temporal factors: meteorology and spatial
                factors: POIs density, with shape `(batch, output_steps, external_dim)`. All features have to be
                time series.
            targets (np.ndarray): Target sensor's labels, with shape `(batch, output_steps, output_dim)`.
        """
        feed_dict = {'local_features': local_features, 'global_features': global_features,
                     'local_attn_states': local_attn_states, 'global_attn_states': global_attn_states,
                     'external_features': external_features, 'targets': targets}
        return feed_dict


def input_transform(local_features,
                    global_features,
                    external_features,
                    targets):
    """Split the model's inputs from matrices to lists on timesteps axis."""
    local_features = split_timesteps(local_features)
    global_features = split_timesteps(global_features)
    external_features = split_timesteps(external_features)
    targets = split_timesteps(targets)
    decoder_inputs = [tf.zeros_like(targets[0], dtype=tf.float32)] + targets[:-1]  # useless when loop func is employed
    return local_features, global_features, external_features, targets, decoder_inputs


def split_timesteps(inputs):
    """Split the input matrix from (batch, timesteps, input_dim) to a step list ([[batch, input_dim], ..., ])."""
    timesteps = inputs.get_shape()[1].value
    feature_dims = inputs.get_shape()[2].value
    inputs = tf.transpose(inputs, [1, 0, 2])
    inputs = tf.reshape(inputs, [-1, feature_dims])
    inputs = tf.split(inputs, timesteps, 0)
    return inputs



