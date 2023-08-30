import numpy as np
from abc import ABC, abstractmethod


class Normalizer(ABC):
    """
    Normalizer is the base abstract class for many normalizers such as MaxMinNormalizer and ZscoreNormalizer.You can also build your own normalizer by inheriting this class.

    Args:
        X(ndarray): Data which normalizer extracts characteristics from.
    """
    @abstractmethod
    def __init__(self, X):
        pass
    @abstractmethod
    def transform(self, X_in):
        pass
    @abstractmethod
    def inverse_transform(self, X_in):
        pass



class MaxMinNormalizer(Normalizer):
    '''
    This class can help normalize and denormalize data using maximum and minimum of data by calling transform and inverse_transform method.

    Args:
        X(ndarray): Data which normalizer extracts characteristics from.
        method(str): Parameter to choose in which way the input data will be processed.
    '''
    def __init__(self, X,method='all'):
        self.method = method
        self._min = np.min(X)
        self._max = np.max(X)
        self._min_by_column = np.min(X,axis=0)
        self._max_by_column = np.max(X,axis=0)

    def transform(self, X):
        '''
        Process input data to obtain normalized data.

        Args:
            X(ndarray): input data. 
        :return: normalized data.
        :type: numpy.ndarray.
        '''
        if self.method=='all':
            return (X - self._min) / (self._max - self._min)
        elif self.method=='column':
            return (X - self._min_by_column) / (self._max_by_column - self._min_by_column)

    def inverse_transform(self, X):
        '''
        Restore normalized data.

        Args:
            X(ndarray): normalized data. 
        :return: denormalized data.
        :type: numpy.ndarray.
        '''
        if self.method=='all':
            return X * (self._max - self._min) + self._min
        elif self.method=='column':
            return X * (self._max_by_column - self._min_by_column) + self._min_by_column


class WhiteNormalizer(Normalizer):
    '''
    This class's normalization won't do anything.
    '''
    def __init__(self, X,method='all'):
        pass

    def transform(self, X):
        '''
        Process input data to obtain normalized data.

        Args:
            X(ndarray): input data. 
        :return: normalized data.
        :type: numpy.ndarray.
        '''
        return X

    def inverse_transform(self, X):
        '''
        Restore normalized data.

        Args:
            X(ndarray): normalized data. 
        :return: denormalized data.
        :type: numpy.ndarray.
        '''
        return X

class ZscoreNormalizer(Normalizer):
    '''
    This class can help normalize and denormalize data using mean and standard deviation in data by calling transform and inverse_transform method.

    Args:
        X(ndarray): Data which normalizer extracts characteristics from.
        method(str): Parameter to choose in which way the input data will be processed.
    '''
    def __init__(self, X,method='all'):
        self.method = method
        self._mean = np.mean(X)
        self._std = np.std(X)
        self._mean_by_column = np.mean(X,axis=0)
        self._std_by_column = np.std(X,axis=0)

    def transform(self, X):
        '''
        Process input data to obtain normalized data.

        Args:
            X(ndarray): input data. 
        :return: normalized data.
        :type: numpy.ndarray.
        '''
        if self.method=='all':
            return (X - self._mean) / self._std
        elif self.method=='column':
            return (X - self._mean_by_column) / self._std_by_column

    def inverse_transform(self, X):
        '''
        Restore normalized data.

        Args:
            X(ndarray): normalized data. 
        :return: denormalized data.
        :type: numpy.ndarray.
        '''
        if self.method=='all':
            return X * self._std + self._mean
        elif self.method=='column':
            return X * self._std_by_column + self._mean_by_column


class MoveSample(object):
    def __init__(self, feature_step, feature_stride, feature_length, target_length):
        self.feature_step = feature_step
        self.feature_stride = feature_stride
        self.feature_length = feature_length
        self.target_length = target_length

    def general_move_sample(self, data):
        feature = []
        target = []
        # sample_num = len(data) - window_size + 1
        # window_size = feature_length + (feature_step-1)*feature_stride + target_length
        for i in range(len(data) - self.feature_length -
                       (self.feature_step-1)*self.feature_stride - self.target_length + 1):
            feature.append([data[i + step*self.feature_stride: i + step*self.feature_stride + self.feature_length]
                            for step in range(self.feature_step)])
            target.append(data[i + (self.feature_step-1) * self.feature_stride + self.feature_length:\
                               i + (self.feature_step-1) * self.feature_stride + self.feature_length + self.target_length])

        return np.array(feature), np.array(target)

class ST_MoveSample(object):
    '''
    This class can converts raw data into temporal features including closenss, period and trend features.

    Args:
        closeness_len(int):The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history.
        period_len(int):The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history.
        trend_len(int):The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history.
        target_length(int):The numbers of steps that need prediction by one piece of history data. Have to be 1 now.
            Default: 1 default:1.
        daily_slots(int): The number of records of one day. Calculated by 24 * 60 /time_fitness. default:24.
    '''
    def __init__(self, closeness_len, period_len, trend_len, target_length=1, daily_slots=24):
        self._c_t = closeness_len
        self._p_t = period_len
        self._t_t = trend_len
        self._target_length = target_length
        self._daily_slots = daily_slots

        # 1 init Move_Sample object
        self.move_sample_closeness = MoveSample(feature_step=self._c_t, feature_stride=1,
                                                feature_length=1, target_length=self._target_length)

        self.move_sample_period = MoveSample(feature_step=self._p_t + 1, feature_stride=int(self._daily_slots),
                                             feature_length=1, target_length=0)

        self.move_sample_trend = MoveSample(feature_step=self._t_t + 1, feature_stride=int(self._daily_slots) * 7,
                                            feature_length=1, target_length=0)

    def move_sample(self, data):
        '''
        Input data to generate closeness, period, trend features and target vector y.

        Args:
            data(ndarray):Orginal temporal data.
        :return:closeness, period, trend and y matrices.
        :type: numpy.ndarray.
        '''
        # 2 general move sample
        closeness, y = self.move_sample_closeness.general_move_sample(data)
        period, _ = self.move_sample_period.general_move_sample(data)
        trend, _ = self.move_sample_trend.general_move_sample(data)

        # 3 remove the front part
        min_length = min(len(closeness), len(period), len(trend))
        closeness = closeness[-min_length:]
        y = y[-min_length:]
        period = period[-min_length:]
        trend = trend[-min_length:]

        # 4 remove tail of period and trend
        period = period[:, :-1]
        trend = trend[:, :-1]

        if self._c_t and self._c_t > 0:
            closeness = np.transpose(closeness, [0] + list(range(3, len(closeness.shape))) + [1, 2])
        else:
            closeness = np.array([])

        if self._p_t and self._p_t > 0:
            period = np.transpose(period, [0] + list(range(3, len(period.shape))) + [1, 2])
        else:
            period = np.zeros(shape=[min_length,closeness.shape[1],0,1])

        if self._t_t and self._t_t > 0:
            trend = np.transpose(trend, [0] + list(range(3, len(trend.shape))) + [1, 2])
        else:
            trend = np.zeros(shape=[min_length,closeness.shape[1],0,1])

        y = np.transpose(y, [0] + list(range(2, len(y.shape))) + [1])

        return closeness, period, trend, y


class SplitData(object):
    '''
    This class can help split data by calling split_data and split_feed_dict method.
    '''
    @staticmethod
    def split_data(data, ratio_list):
        '''
        Divide the data based on the given parameter ratio_list.
        
        Args:
            data(ndarray):Data to be split.
            ratio_list(list):Split ratio, the `data` will be split according to the ratio.
        :return:The elements in the returned list are the divided data, and the 
            dimensions of the list are the same as ratio_list.
        :type: list
        '''
        if np.sum(ratio_list) != 1:
            ratio_list = np.array(ratio_list)
            ratio_list = ratio_list / np.sum(ratio_list)
        return [data[int(sum(ratio_list[0:e])*len(data)):
                     int(sum(ratio_list[0:e+1])*len(data))] for e in range(len(ratio_list))]

    @staticmethod
    def split_feed_dict(feed_dict, sequence_length, ratio_list):
        '''
        Divide the `value` data in `feed_dict` based on the given parameter ratio_list.

        Args:
            feed_dict(dict):It is a dictionary composed of `key-value` pairs.
            sequence_length(int):If the length of `value` in `feed_dict` is equal to sequence_length, 
                then this method divides the `value` according to the ratio without changing its `key`.
            ratio_list(list):Split ratio, the data will be split according to the ratio.
        :return: The elements in the returned list are divided dictionaries, and the dimensions of the list are the same as ratio_list.
        :type: list
        '''
        if np.sum(ratio_list) != 1:
            ratio_list = np.array(ratio_list)
            ratio_list = ratio_list / np.sum(ratio_list)

        return [{key: value[int(sum(ratio_list[0:e])*len(value)):int(sum(ratio_list[0:e+1])*len(value))]
                 if len(value) == sequence_length else value for key, value in feed_dict.items()}
                for e in range(len(ratio_list))]

def chooseNormalizer(in_arg,X_train):
    '''
    Choose a proper normalizer consistent with user's input.

    Args:
        in_arg(str|bool|object):Function is based on it to choose different normalizer.
        X_train(numpy.ndarray):Function is based on it to initialize the normalizer.
    :return: The normalizer consistent with definition.
    :type: object.
    '''
    if type(in_arg) == str:
        if '-' in in_arg:
            method,way=in_arg.split('-')
            if method=='Zscore' or method=='zscore' or method=='ZScore':
                return ZscoreNormalizer(X_train,way)
            elif method=='MaxMin' or method=='maxmin' or method=='Maxmin' or method=='MinMax' or method=='Minmax' or method=='minmax':
                return MaxMinNormalizer(X_train,way)
            else:
                raise ValueError('We havn\'t support thie method for normalization yet')
        else:
            raise ValueError('We don\'t accept this format of str input for how to do normalization')
    elif type(in_arg) == bool:
        if in_arg:
            return MaxMinNormalizer(X_train)
        else:
            return WhiteNormalizer(X_train)
    elif type(in_arg) == object:
        if hasattr(in_arg,'transform') and hasattr(in_arg,'inverss_transform'):
            return in_arg(X_train)
        else:
            raise TypeError('Your custom normalizer is not in compliance')
    else:
        raise TypeError('We don\'t accept {} of input for how to do normalization')

if __name__ == '__main__':
    a = np.arange(0,1000)
    st_movesample = ST_MoveSample(6,7,4)
    closeness,period,trend,y = st_movesample.move_sample(a)
    print(y)

