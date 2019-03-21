import numpy as np


class Accuracy(object):
    def __init__(self):
        pass

    @staticmethod
    def RMSE(prediction, target, **kwargs):
        threshold = None
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        if threshold is None:
            return np.sqrt(np.mean(np.square(prediction - target)))
        else:
            return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                                  target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]

    @staticmethod
    def MAPE(prediction, target, **kwargs):
        threshold = 0
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        return (np.dot((np.abs(prediction - target) / (target + (1 - (target > threshold)))).reshape([1, -1]),
                       target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]

    @staticmethod
    def RMSE_Grid(prediction, target, **kwargs):
        threshold = None
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        if threshold is None:
            return np.sqrt(np.mean(np.square(prediction - target), axis=0))
        else:
            return np.sqrt(np.sum(np.square(prediction - target) *
                                  (target > threshold), axis=0) / np.sum(target > threshold, axis=0))

    @staticmethod
    def MAPE_Grid(prediction, target, **kwargs):
        threshold = 0
        for key, value in kwargs.items():
            if key.lower() == 'threshold':
                threshold = value
        return np.sum((np.abs(prediction - target) / (target + (1 - (target > threshold))))
                      * (target > threshold), axis=0) / np.sum(target > threshold, axis=0)