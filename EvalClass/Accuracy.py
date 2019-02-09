import numpy as np


class Accuracy(object):
    def __init__(self):
        pass

    @staticmethod
    def RMSE(p, t, **kwargs):
        threshold = None
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        if threshold is None:
            return np.sqrt(np.mean(np.square(p - t)))
        else:
            return np.sqrt(np.dot(np.square(p - t).reshape([1, -1]),
                                  t.reshape([-1, 1]) > threshold) / np.sum(t > threshold))[0][0]

    @staticmethod
    def MAPE(p, t, **kwargs):
        threshold = 0
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        return (np.dot((np.abs(p - t) / (t + (1 - (t > threshold)))).reshape([1, -1]),
                      t.reshape([-1, 1]) > threshold) / np.sum(t > threshold))[0, 0]

    @staticmethod
    def RMSE_Grid(p, t, **kwargs):
        threshold = None
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        if threshold is None:
            return np.sqrt(np.mean(np.square(p - t), axis=0))
        else:
            return np.sqrt(np.sum(np.square(p - t) * (t > threshold), axis=0) / np.sum(t > threshold, axis=0))

    @staticmethod
    def MAPE_Grid(p, t, **kwargs):
        threshold = 0
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        return np.sum((np.abs(p - t) / (t + (1 - (t > threshold)))) * (t > threshold), axis=0) / np.sum(t > threshold, axis=0)