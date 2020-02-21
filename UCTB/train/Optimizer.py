import keras
import tensorflow as tf
import yaml


class Optimizer(object):
    def __init__(self, decay_param=None, lr=None):
        # if not specified, naive method
        if decay_param is None:
            self._decay_func = None
            self._decay_param = {"lr": lr}
        else:
            # Parse params
            args = {}
            with open(decay_param, 'r') as f:
                args.update(yaml.load(f))
            self._decay_func = eval(args['decay_func'])
            self._decay_param = args

    def build(self, loss_pre):
        '''
        this func return train_op tensor.
        '''
        global_step = tf.Variable(0, trainable=False, name="global_step")
        try:
            if self._decay_func is None:
                learning_rate = tf.Variable(tf.constant(
                    self._decay_param['lr'], dtype=tf.float32))
            elif self._decay_func is tf.train.exponential_decay:
                learning_rate = self._decay_func(self._decay_param['starter_learning_rate'], global_step,
                                                 self._decay_param['decay_steps'], self._decay_param['decay_rate'], staircase=self._decay_param['staircase'])
            elif self._decay_func is tf.train.cosine_decay_restarts:
                learning_rate = self._decay_func(self._decay_param['learning_rate'], global_step,
                                                 self._decay_param['first_decay_steps'], t_mul=self._decay_param['t_mul'], m_mul=self._decay_param['m_mul'], alpha=self._decay_param['alpha'])
            else:
                raise KeyError(
                    "decay_func is not defined, see the doc for help.")
            return tf.train.AdamOptimizer(learning_rate).minimize(loss_pre, name='train_op'), global_step.name, learning_rate.name
        except Exception as e:
            raise KeyError("Decay learning param error. Check param files.\n"+e)
