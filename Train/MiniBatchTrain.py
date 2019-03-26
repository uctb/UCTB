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
            index = [self.__sample_size-self.__batch_size, self.__sample_size]
            self.__batch_counter = 0

        return [np.array(e[index[0]: index[1]]) for e in self.__data]

    def restart(self):
        self.__batch_counter = 0


class MiniBatchFeedDict(object):

    def __init__(self, feed_dict, sequence_length, batch_size, shuffle=True):

        self._sequence_length = sequence_length
        self._batch_size = batch_size

        self._dynamic_data_names = []
        self._dynamic_data_values = []

        self._batch_dict = {}

        for key, value in feed_dict.items():
            if len(value) == sequence_length:
                self._dynamic_data_names.append(key)
                self._dynamic_data_values.append(value)
            else:
                self._batch_dict[key] = value

        if shuffle:
            self._dynamic_data_values = MiniBatchFeedDict.shuffle(self._dynamic_data_values)

        self._batch_counter = 0

        self.num_batch = int(self._sequence_length / self._batch_size) \
            if self._sequence_length % self._batch_size == 0 else int(self._sequence_length / self._batch_size) + 1

    def get_batch(self):
        if self._batch_counter + self._batch_size <= self._sequence_length:
            index = [self._batch_counter, self._batch_counter + self._batch_size]
            self._batch_counter += self._batch_size
        else:
            index = [self._sequence_length-self._batch_size, self._sequence_length]
            self._batch_counter = 0

        for i in range(len(self._dynamic_data_names)):
            key = self._dynamic_data_names[i]
            self._batch_dict[key] = np.array(self._dynamic_data_values[i][index[0]:index[1]])

        return self._batch_dict

    @staticmethod
    def shuffle(data):
        middle = list(zip(*data))
        np.random.shuffle(middle)
        return list(zip(*middle))

    def restart(self):
        self._batch_counter = 0
