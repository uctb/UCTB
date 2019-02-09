import numpy as np


class MiniBatchTrain():

    def __init__(self, X, Y, batch_size):
        # The first dimension of X should be sample size
        # The first dimension of Y should be sample size

        self.__X, self.__Y = self.shuffle(X, Y)

        self.__sample_size = len(X)

        self.__batch_counter = 0
        self.__batch_size = batch_size

        self.num_batch = int(self.__sample_size / self.__batch_size) \
        if self.__sample_size % self.__batch_size == 0 else int(self.__sample_size / self.__batch_size) + 1

    @staticmethod
    def shuffle(X, Y):
        xy = list(zip(X, Y))
        np.random.shuffle(xy)
        return np.array([e[0] for e in xy], dtype=np.float32), np.array([e[1] for e in xy], dtype=np.float32)

    def get_batch(self):
        if self.__batch_counter + self.__batch_size <= self.__sample_size:
            batch_x = self.__X[self.__batch_counter: self.__batch_counter + self.__batch_size]
            batch_y = self.__Y[self.__batch_counter: self.__batch_counter + self.__batch_size]
            self.__batch_counter = self.__batch_counter + self.__batch_size
        else:
            batch_x = self.__X[-self.__batch_size: ]
            batch_y = self.__Y[-self.__batch_size: ]
            self.__batch_counter = 0

        return batch_x, batch_y

    def restart(self):
        self.__batch_counter = 0


class MiniBatchTrainMultiData(object):

    def __init__(self, data, batch_size, shuffle=True):
        if shuffle:
            self.__data = self.shuffle(data)
        else:
            self.__data = data

        self.__sample_size = len(self.__data[0])

        self.__batch_counter = 0
        self.__batch_size = batch_size

        self.num_batch = int(self.__sample_size / self.__batch_size) \
        if self.__sample_size % self.__batch_size == 0 else int(self.__sample_size / self.__batch_size) + 1

    @staticmethod
    def shuffle(data):
        middle = list(zip(*data))
        np.random.shuffle(middle)
        return list(zip(*middle))

    def get_batch(self):
        if self.__batch_counter + self.__batch_size <= self.__sample_size:
            index = [self.__batch_counter, self.__batch_counter + self.__batch_size]
            self.__batch_counter = self.__batch_counter + self.__batch_size
        else:
            index = [-self.__batch_size, self.__sample_size]
            self.__batch_counter = 0

        return [np.array(e[index[0]: index[1]]) for e in self.__data]

    def restart(self):
        self.__batch_counter = 0