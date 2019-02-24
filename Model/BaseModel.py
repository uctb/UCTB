import os
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
        self._summary = {}
        self._summary_writer = tf.summary.FileWriter(self._log_dir)

        # TF Session
        self._GPU_DEVICE = GPU_DEVICE
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU_DEVICE
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=self._graph, config=self._config)

    def add_summary(self, scalar_dict):
        with self._graph.as_default():
            for scalar_name, scalar_tensor_name in scalar_dict.items():
                self._summary[scalar_name] = tf.summary.scalar(scalar_name,
                                                               self._graph.get_tensor_by_name(scalar_tensor_name)).name

    def _run(self, input_dict, output_keys=None, op_keys=None, summary_keys=None):
        feed_dict = {}
        for name, value in input_dict.items():
            if value is not None:
                feed_dict[self._graph.get_tensor_by_name(self._input[name])] = value

        execute_output_names = output_keys if output_keys else list(self._output.keys())
        execute_op_names = op_keys if op_keys else list(self._op.keys())
        execute_summary_names = summary_keys if summary_keys else list(self._summary.keys())

        output_tensor_list = [self._graph.get_tensor_by_name(self._output[name]) for name in execute_output_names]
        output_tensor_list += [self._graph.get_operation_by_name(self._op[name]) for name in execute_op_names]
        output_tensor_list += [self._graph.get_tensor_by_name(self._summary[name]) for name in execute_summary_names]

        outputs = self._session.run(output_tensor_list, feed_dict=feed_dict)

        for i in range(1, len(execute_summary_names)+1):
            self._summary_writer.add_summary(outputs[-i])

        return {execute_output_names[i]: outputs[i] for i in range(len(execute_output_names))}

    def fit(self, input_dict, output_keys=None, op_keys=None, summary_keys=None):
        return self._run(input_dict, output_keys, op_keys, summary_keys)

    def predict(self, input_dict, output_keys=None):
        return self._run(input_dict, output_keys, op_keys=[], summary_keys=[])
    
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