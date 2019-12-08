import numpy as np


class MiniBatchTrain():
    '''
    Get small batches of data for training at once.

    Args:
        X(ndarray):Input features. The first dimension of X should be sample size.
        Y(ndarray):Target values. The first dimension of Y should be sample size.
        batch_size(int): The number of data for one training session.
    '''
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
        '''
        Input (X, Y) pairs, shuffle and return it.
        '''
        xy = list(zip(X, Y))
        np.random.shuffle(xy)
        return np.array([e[0] for e in xy], dtype=np.float32), np.array([e[1] for e in xy], dtype=np.float32)

    def get_batch(self):
        '''
        Returns a batch of X, Y pairs each time. There are internal variables 
        to record the number of batches currently generated. When the last data 
        is not enough to generate a batch, a batch of data from the tail is returned.
        '''
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
        '''
        Set the variable that records the number of batches currently generated to 0, so that 
        we can call the `get_batch` method to generate training data in batches from scratch.
        '''
        self.__batch_counter = 0


class MiniBatchTrainMultiData(object):
    '''
    Get small batches of data for training at once.

    Args:
        data(ndarray): Input data. Its first dimension should be sample size.
        batch_size(int): The number of data for one training session.
        shuffle(bool): If set `True`, the input data will be shuffled. default:True.
    '''
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
        '''
        Returns a batch of data each time. There are internal variables 
        to record the number of batches currently generated. When the last data 
        is not enough to generate a batch, a batch of data from the tail is returned.
        '''
        if self.__batch_counter + self.__batch_size <= self.__sample_size:
            index = [self.__batch_counter, self.__batch_counter + self.__batch_size]
            self.__batch_counter = self.__batch_counter + self.__batch_size
        else:
            index = [self.__sample_size-self.__batch_size, self.__sample_size]
            self.__batch_counter = 0

        return [np.array(e[index[0]: index[1]]) for e in self.__data]

    def restart(self):
        '''
        Set the variable that records the number of batches currently generated to 0, so that 
        we can call the `get_batch` method to generate training data in batches from scratch.
        '''
        self.__batch_counter = 0


class MiniBatchFeedDict(object):
    '''
    Get small batches of data from dict for training at once.

    Args:
        feed_dict(dict): Data dictionary consisting of key-value pairs.
        sequence_length(int): Only divide value in `feed_dict` whose length is equal 
            to `sequence_length` into several batches.
        batch_size(int): The number of data for one training session.
        shuffle(bool): If set `True`, the input dict will be shuffled. default:True.
    '''
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
        '''
        For the `value` in `feed_dict` whose length is equal to sequence_length, divide the `value` 
        into several batches, and return one batch in order each time. For those whose length is not 
        equal to sequence_length, do not change `value`and return it directly. There are internal variables 
        to record the number of batches currently generated. When the last data is not enough to 
        generate a batch, a batch of data from the tail is returned.
        '''
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
        '''
        Set the variable that records the number of batches currently generated to 0, so that 
        we can call the `get_batch` method to generate training data in batches from scratch.
        '''
        self._batch_counter = 0
