import os
import numpy as np
import shutil
import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator

from ..train.MiniBatchTrain import MiniBatchFeedDict
from ..preprocess.preprocessor import SplitData
from ..train.EarlyStopping import *


class BaseModel(object):

    def __init__(self, code_version, model_dir, gpu_device):

        # model input and output
        self._input = {}
        self._output = {}
        self._op = {}
        self._variable_init = None
        self._saver = None

        self._code_version = code_version
        self._model_dir = model_dir

        # TF Graph
        self._graph = tf.Graph()

        self._converged = False
        self._log_dir = os.path.join(self._model_dir, self._code_version)
        self._global_step = 0
        self._summary = None
        self._summary_writer = tf.summary.FileWriter(self._log_dir)

        self.trainable_vars = 0

        # TF Session
        self._GPU_DEVICE = gpu_device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU_DEVICE
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=self._graph, config=self._config)

    def build(self):
        with self._graph.as_default():
            ####################################################################
            # Add summary, variable_init and summary
            # The variable name of them are fixed
            self.trainable_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            self._saver = tf.train.Saver()
            self._variable_init = tf.global_variables_initializer()
            self._summary = self._summary_histogram().name
            ####################################################################
        self._session.run(self._variable_init)

    def add_summary(self, name, value, global_step):
        value_record = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self._summary_writer.add_summary(value_record, global_step)

    def _summary_histogram(self):
        with self._graph.as_default():
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        self._summary_writer.add_graph(self._graph)
        return tf.summary.merge_all()

    def _run(self, feed_dict, output_names, op_names):
        feed_dict_tf = {}
        for name, value in feed_dict.items():
            if value is not None:
                feed_dict_tf[self._graph.get_tensor_by_name(self._input[name])] = value

        output_tensor_list = [self._graph.get_tensor_by_name(self._output[name]) for name in output_names]
        output_tensor_list += [self._graph.get_operation_by_name(self._op[name]) for name in op_names]

        outputs = self._session.run(output_tensor_list, feed_dict=feed_dict_tf)

        return {output_names[i]: outputs[i] for i in range(len(output_names))}

    def _get_feed_dict(self, **kwargs):
        return kwargs

    def fit(self, sequence_length, output_names=('loss', ), op_names=('train_op', ), evaluate_loss_name='loss',
            batch_size=64, max_epoch=10000, validate_ratio=0.1, shuffle_data=True,
            early_stop_method='t-test', early_stop_length=10, early_stop_patience=0.1,
            verbose=True, save_model=True, **kwargs):

        """
        :param sequence_length: int, the sequence length which is use in mini-batch training
        :param output_names: list, [output_tensor1_name, output_tensor1_name, ...]
        :param op_names: list, [operation1_name, operation2_name, ...]
        :param evaluate_loss_name: str, should be on of the output_names, evaluate_loss_name was use in
                                   early-stopping
        :param batch_size: int, default 64, batch size
        :param max_epoch: int, default 10000, max number of epochs
        :param validate_ratio: float, default 0.1, the ration of data that will be used as validation dataset
        :param shuffle_data: bool, default True, whether shuffle data in mini-batch train
        :param early_stop_method: should be 't-test' or 'naive', both method are explained in train.EarlyStopping
        :param early_stop_length: int, must provide when early_stop_method='t-test'
        :param early_stop_patience: int, must provide when early_stop_method='naive'
        :param verbose: Bool, flag to print training information or not
        :param save_model: Bool, flog to save model or not
        """

        try:
            self.load(self._code_version)
            print('Found model in disk')
            if self._converged:
                print('Model converged, stop training')
                return
            else:
                print('Model not converged, continue at step', self._global_step)
                start_epoch = self._global_step
        except FileNotFoundError:
            print('No model found, start training')
            start_epoch = 0

        if not 0 < validate_ratio < 1:
            raise ValueError('validate_ratio should between (0, 1), given', validate_ratio)

        if evaluate_loss_name not in output_names:
            raise ValueError('evaluate_loss_name not shown in', output_names)

        if len(op_names) == 0:
            raise ValueError('No operation given')
        else:
            print('Running Operation', op_names)

        # Get feed_dict
        feed_dict = self._get_feed_dict(**kwargs)

        # Split data into train-data and validation data
        train_feed_dict, val_feed_dict = SplitData.split_feed_dict(feed_dict,
                                                                   sequence_length=sequence_length,
                                                                   ratio_list=[1 - validate_ratio, validate_ratio])

        train_sequence_length = int(sequence_length*(1-validate_ratio))
        val_sequence_len = sequence_length - train_sequence_length

        # build mini-batch data source on train-data
        train_dict_mini_batch = MiniBatchFeedDict(feed_dict=train_feed_dict,
                                                  sequence_length=train_sequence_length,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_data)

        # record the best result of "evaluate_loss_name"
        best_record = None
        # init early stopping object
        if early_stop_method.lower() == 't-test':
            early_stop = EarlyStoppingTTest(length=early_stop_length, p_value_threshold=early_stop_patience)
        else:
            early_stop = EarlyStopping(patience=early_stop_patience)

        # start mini-batch training
        for epoch in range(start_epoch, max_epoch):
            train_output_list = []
            for i in range(train_dict_mini_batch.num_batch):
                # train
                train_output = self._run(feed_dict=train_dict_mini_batch.get_batch(),
                                         output_names=output_names,
                                         op_names=op_names)
                train_output_list.append(train_output)

            # validation
            val_output = self.predict(**val_feed_dict, output_names=output_names,
                                      sequence_length=val_sequence_len,
                                      cache_volume=batch_size)

            # Here we only care about the evaluate_loss_value
            evaluate_loss_value = np.mean(val_output[evaluate_loss_name])

            # Add Summary
            for name in output_names:
                self.add_summary(name='train_' + name, value=np.mean([e[name] for e in train_output_list]),
                                 global_step=epoch)
                self.add_summary(name='val_' + name, value=evaluate_loss_value, global_step=epoch)
                # print training messages
                if verbose:
                    print('Epoch %s:' % epoch,
                          'train_' + name, np.mean([e[name] for e in train_output_list]),
                          'val_' + name, evaluate_loss_value)

            # manual_summary the histograms
            self.manual_summary(global_step=epoch)

            if early_stop.stop(evaluate_loss_value):
                if save_model:
                    self._log('Converged')
                break

            # save the model if evaluate_loss_value is smaller than best_record
            if (best_record is None or evaluate_loss_value < best_record) and save_model:
                best_record = evaluate_loss_value
                self.save(self._code_version, epoch)

    def predict(self, output_names, sequence_length, cache_volume=64, **kwargs):

        '''
        :param output_names: list, [output_tensor_name1, output_tensor_name2, ...]
        :param sequence_length: int, the length of sequence, which is use in mini-batch training
        :param cache_volume: int, default 64, we need to set cache_volume if the cache can not hold
                             the whole validation dataset
        :return: outputs_dict: dict, like {output_tensor1_name: output_tensor1_value, ...}
        '''

        # Get feed_dict
        feed_dict = self._get_feed_dict(**kwargs)

        if cache_volume and sequence_length:
            # storing the prediction result
            outputs_list = []
            outputs_dict = {}
            for i in range(0, sequence_length, cache_volume):
                tmp_output = self._run({key: value[i:i+cache_volume] if len(value) == sequence_length else value
                                        for key, value in feed_dict.items()},
                                       output_names, op_names=[])
                outputs_list.append(tmp_output)
            # stack the output together
            for key in outputs_list[0]:
                outputs_dict[key] = np.vstack([e[key] for e in outputs_list])
        else:
            outputs_dict = self._run(feed_dict, output_names, op_names=[])

        return outputs_dict

    def manual_summary(self, global_step=None):
        self._summary_writer.add_summary(self._session.run(self._graph.get_tensor_by_name(self._summary)),
                                         global_step=global_step)

    def _log(self, text):
        save_dir_subscript = os.path.join(self._log_dir, self._code_version)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        with open(os.path.join(save_dir_subscript, 'log.txt'), 'a+', encoding='utf-8') as f:
            f.write(text + '\n')

    def _get_log(self):
        save_dir_subscript = os.path.join(self._log_dir, self._code_version)
        if os.path.isfile(os.path.join(save_dir_subscript, 'log.txt')):
            with open(os.path.join(save_dir_subscript, 'log.txt'), 'r', encoding='utf-8') as f:
                return [e.strip('\n') for e in f.readlines()]
        else:
            return []

    def save(self, subscript, global_step):
        save_dir_subscript = os.path.join(self._log_dir, subscript)
        # delete if exist
        # if os.path.isdir(save_dir_subscript):
        #     shutil.rmtree(save_dir_subscript, ignore_errors=True)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        self._saver.save(sess=self._session, save_path=os.path.join(save_dir_subscript, subscript),
                         global_step=global_step)

    def load(self, subscript):
        save_dir_subscript = os.path.join(self._log_dir, subscript)
        if len(os.listdir(save_dir_subscript)) == 0:
            print('model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            meta_file = [e for e in os.listdir(save_dir_subscript) if e.startswith(subscript) and e.endswith('.meta')]
            self._global_step = max([int(e.split('.')[0].split('-')[-1]) for e in meta_file])
            self._saver.restore(sess=self._session,
                                save_path=os.path.join(save_dir_subscript, subscript + '-%s' % self._global_step))
            self._global_step += 1
            # parse the log-file
            log_list = self._get_log()
            for e in log_list:
                if e.lower() == 'converged':
                    self._converged = True

    def close(self):
        self._session.close()

    def load_event_scalar(self, scalar_name='val_loss'):
        event_files = [e for e in os.listdir(self._log_dir) if e.startswith('events.out')]
        result = []
        for f in event_files:
            ea = event_accumulator.EventAccumulator(os.path.join(self._log_dir, f))
            ea.Reload()
            if scalar_name in ea.scalars.Keys():
                result += [[e.wall_time, e.step, e.value] for e in ea.scalars.Items(scalar_name)]
        return result
