import numpy as np


class Normalize(object):
    def __init__(self, X):
        self.__min = np.min(X)
        self.__max = np.max(X)

    def min_max_normal(self, X):
        return (X - self.__min) / (self.__max - self.__min)

    def min_max_denormal(self, X):
        return X * (self.__max - self.__min) + self.__min

    def white_normal(self):
        pass


class MoveSample(object):
    def __init__(self, feature_step, feature_stride, feature_length, target_length):
        self.feature_step = feature_step
        self.feature_stride = feature_stride
        self.feature_length = feature_length
        self.target_length = target_length

    def general_move_sample(self, data):
        feature = []
        target = []
        for i in range(len(data) - self.feature_length - (self.feature_step-1) * self.feature_stride - self.target_length + 1):
            feature.append([data[i + step*self.feature_stride: i + step*self.feature_stride + self.feature_length]
                            for step in range(self.feature_step)])
            target.append(data[i + (self.feature_step-1) * self.feature_stride + self.feature_length:\
                               i + (self.feature_step-1) * self.feature_stride + self.feature_length + self.target_length])

        return np.array(feature), np.array(target)


class SplitData(object):
    @staticmethod
    def split_data(data, train, val, test):
        return data[:int(len(data) * train)], \
               data[int(len(data) * train): int(len(data) * (train + val))], \
               data[-int(len(data) * test):]