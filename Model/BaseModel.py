import os
import numpy as np
import shutil
import tensorflow as tf


class BaseModel(object):

    def __init__(self, code_version, model_dir, GPU_DEVICE):

        # Model input and output
        self._input = {}
        self._output = {}
        self._op = {}
        self._variable_init = None
        self._saver = None

        self.__build = True

        self._code_version = code_version
        self._model_dir = model_dir

        # TF Graph
        self._graph = tf.Graph()

        self._log_dir =  os.path.join(self._model_dir, self._code_version)
        self._summary = None
        self._summary_writer = tf.summary.FileWriter(self._log_dir)

        # TF Session
        self._GPU_DEVICE = GPU_DEVICE
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU_DEVICE
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=self._graph, config=self._config)

    def add_summary(self, name, value, global_step):
        value_record = tf.Summary(
            value=[tf.Summary.Value(tag=name, simple_value=value)])
        self._summary_writer.add_summary(value_record, global_step)

    def _summary_histogram(self):
        with self._graph.as_default():
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        return tf.summary.merge_all()

    def _run(self, input_dict, output_keys=None, op_keys=None, global_step=None, summary=True):
        feed_dict = {}
        for name, value in input_dict.items():
            if value is not None:
                feed_dict[self._graph.get_tensor_by_name(self._input[name])] = value

        execute_output_names = output_keys if output_keys else list(self._output.keys())
        execute_op_names = op_keys if op_keys else list(self._op.keys())

        output_tensor_list = [self._graph.get_tensor_by_name(self._output[name]) for name in execute_output_names]
        output_tensor_list += [self._graph.get_operation_by_name(self._op[name]) for name in execute_op_names]
        output_tensor_list += [self._graph.get_tensor_by_name(self._summary)]

        outputs = self._session.run(output_tensor_list, feed_dict=feed_dict)

        if summary:
            self._summary_writer.add_summary(outputs[-1], global_step=global_step)

        return {execute_output_names[i]: outputs[i] for i in range(len(execute_output_names))}

    def fit(self, input_dict, output_keys=None, op_keys=None, **kwargs):
        return self._run(input_dict, output_keys, op_keys, **kwargs)

    def predict(self, input_dict, output_keys=None, cache_volume=None, sequence_length=None):
        if cache_volume and sequence_length:
            # storing the prediction result
            outputs_list = []
            outputs_dict = {}
            for i in range(0, sequence_length, cache_volume):
                tmp_output = self._run({key:value[i:i+cache_volume] if len(value) == sequence_length else value
                                    for key, value in input_dict.items()},
                                   output_keys, op_keys=[])
                outputs_list.append(tmp_output)
            # stack the output together
            for key in outputs_list[0]:
                outputs_dict[key] = np.vstack([e[key] for e in outputs_list])
        else:
            outputs_dict = self._run(input_dict, output_keys, op_keys=[])

        return outputs_dict

    def evaluate(self, input_dict, target_key, prediction_key, metric, de_normalizer=None,
                 output_keys=None, cache_volume=None, sequence_length=None, **kwargs):

        outputs = self.predict(input_dict, output_keys, cache_volume=cache_volume, sequence_length=sequence_length)

        target = input_dict[target_key]
        prediction = outputs[prediction_key]

        if de_normalizer:
            target = de_normalizer(target)
            prediction = de_normalizer(prediction)

        return [m(prediction, target, **kwargs) for m in metric]

    def manual_summary(self, global_step=None):
        self._summary_writer.add_summary(self._session.run(self._graph.get_tensor_by_name(self._summary)),
                                         global_step=global_step)
    
    def save(self, subscript):
        save_dir_subscript = os.path.join(self._log_dir, subscript)
        # delete if exist
        if os.path.isdir(save_dir_subscript):
            shutil.rmtree(save_dir_subscript, ignore_errors=True)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        self._saver.save(sess=self._session, save_path=os.path.join(save_dir_subscript, subscript))

    def load(self, subscript):
        save_dir_subscript = os.path.join(self._log_dir, subscript)
        if len(os.listdir(save_dir_subscript)) == 0:
            print('Model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            self._saver.restore(self._session, os.path.join(save_dir_subscript, subscript))

    def close(self):
        self._session.close()