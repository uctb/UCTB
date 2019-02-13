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

        # TF Session
        self._GPU_DEVICE = GPU_DEVICE
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU_DEVICE
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=self._graph, config=self._config)

    def save(self, subscript):
        save_dir = os.path.join(self._model_dir, self._code_version)
        save_dir_subscript = os.path.join(save_dir, subscript)
        # delete if exist
        if os.path.isdir(save_dir_subscript):
            shutil.rmtree(save_dir_subscript, ignore_errors=True)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        self._saver.save(sess=self._session, save_path=os.path.join(save_dir_subscript, subscript))

    def load(self, subscript):
        save_dir = os.path.join(self._model_dir, self._code_version)
        save_dir_subscript = os.path.join(save_dir, subscript)
        if len(os.listdir(save_dir_subscript)) == 0:
            print('Model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            self._saver.restore(self._session, os.path.join(save_dir_subscript, subscript))

    def close(self):
        self._session.close()