import numpy as np


class Normalizer(object):
    '''
    This class can help normalize and denormalize data by calling min_max_normal and min_max_denormal method.
    '''
    def __init__(self, X):
        self._min = np.min(X)
        self._max = np.max(X)

    def min_max_normal(self, X):
        '''
        Input X, return normalized results.
        :type: numpy.ndarray
        '''
        return (X - self._min) / (self._max - self._min)

    def min_max_denormal(self, X):
        '''
        Input X, return denormalized results.
        :type: numpy.ndarray
        '''
        return X * (self._max - self._min) + self._min

    # def white_normal(self):
    #     pass


class MoveSample(object):
    def __init__(self, feature_step, feature_stride, feature_length, target_length):
        self.feature_step = feature_step
        self.feature_stride = feature_stride
        self.feature_length = feature_length
        self.target_length = target_length

    def general_move_sample(self, data):
        feature = []
        target = []
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
            period = np.array([])

        if self._t_t and self._t_t > 0:
            trend = np.transpose(trend, [0] + list(range(3, len(trend.shape))) + [1, 2])
        else:
            trend = np.array([])

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