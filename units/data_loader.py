import os
import copy
import datetime
import numpy as np

from dateutil.parser import parse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from UCTB.preprocess import MoveSample, SplitData, ST_MoveSample, Normalizer
from UCTB.dataset.dataset import DataSet


class NodeTrafficLoader(object):
    """Reference:
       https://github.com/uctb/UCTB
       Args:
        dataset (str): A string containing path of the dataset pickle file or a string of name of the dataset.
        city (:obj:`str` or ``None``): ``None`` if dataset is file path, or a string of name of the city.
            Default: ``None``
        data_range: The range of data extracted from ``self.dataset`` to be further used. If set to ``'all'``, all data in
            ``self.dataset`` will be used. If set to a float between 0.0 and 1.0, the relative former proportion of data in
            ``self.dataset`` will be used. If set to a list of two integers ``[start, end]``, the data from *start* day to
            (*end* - 1) day of data in ``self.dataset`` will be used. Default: ``'all'``
        train_data_length: The length of train data. If set to ``'all'``, all data in the split train set will be used.
            If set to int, the latest ``train_data_length`` days of data will be used as train set. Default: ``'all'``
        test_ratio (float): The ratio of test set as data will be split into train set and test set. Default: 0.1
        closeness_len (int): The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history. Default: 6
        period_len (int): The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history. Default: 7
        trend_len (int): The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history. Default: 4
        target_length (int): The numbers of steps that need prediction by one piece of history data. Have to be 1 now.
            Default: 1
        normalize (bool): If ``True``, do min-max normalization on data. Default: ``True``
        with_tpe (bool): If ``True``, data loader will build time position embeddings. Default: ``False``
        data_dir (:obj:`str` or ``None``): The dataset directory. If set to ``None``, a directory will be created. If
            ``dataset`` is file path, ``data_dir`` should be ``None`` too. Default: ``None``
        MergeIndex(int): The granularity of dataset will be ``MergeIndex`` * original granularity.
        MergeWay(str): How to change the data granularity. Now it can be ``sum`` ``average`` or ``max``.
        remove(bool): If ``True``, dataloader  will remove stations whose average traffic is less than 1. 
            Othewise, dataloader will use all stations.

    Attributes:
        dataset (DataSet): The DataSet object storing basic data.
        daily_slots (int): The number of time slots in one single day.
        station_number (int): The number of nodes.
        external_dim (int): The number of dimensions of external features.
        train_closeness (np.ndarray): The closeness history of train set data. When ``with_tpe`` is ``False``,
            its shape is [train_time_slot_num, ``station_number``, ``closeness_len``, 1].
            On the dimension of ``closeness_len``, data are arranged from earlier time slots to later time slots.
            If ``closeness_len`` is set to 0, train_closeness will be an empty ndarray.
            ``train_period``, ``train_trend``, ``test_closeness``, ``test_period``, ``test_trend`` have similar shape
            and construction.
        train_y (np.ndarray): The train set data. Its shape is [train_time_slot_num, ``station_number``, 1].
            ``test_y`` has similar shape and construction.
    """

    def __init__(self,
                 dataset,
                 city=None,
                 data_range='all',
                 train_data_length='all',
                 test_ratio=0.1,
                 closeness_len=6,
                 period_len=7,
                 trend_len=0,
                 target_length=1,
                 step=0,
                 normalize=True,
                 with_tpe=False,
                 data_dir=None,
                 MergeIndex=1,
                 MergeWay="sum",
                 remove=False, **kwargs):

        self.dataset = DataSet(dataset, MergeIndex, MergeWay, city, data_dir=data_dir)

        self.loader_id = "{}_{}_{}_{}_{}_{}_{}_N".format(data_range, train_data_length, test_ratio, closeness_len,
                                                         period_len, trend_len, self.dataset.time_fitness)

        self.daily_slots = int(24 * 60 / self.dataset.time_fitness)

        self.closeness_len = int(closeness_len)
        self.period_len = int(period_len)
        self.trend_len = int(trend_len)

        assert type(self.closeness_len) is int and self.closeness_len >= 0
        assert type(self.period_len) is int and self.period_len >= 0
        assert type(self.trend_len) is int and self.trend_len >= 0

        if type(data_range) is str and data_range.lower().startswith("0."):
            data_range = float(data_range)
        if type(data_range) is str and data_range.lower() == 'all':
            data_range = [0, len(self.dataset.node_traffic)]
        elif type(data_range) is float:
            data_range = [0, int(data_range * len(self.dataset.node_traffic))]
        else:
            data_range = [int(data_range[0] * self.daily_slots), int(data_range[1] * self.daily_slots)]

        num_time_slots = data_range[1] - data_range[0]

        # traffic feature
        if remove:
            self.traffic_data_index = np.where(np.mean(self.dataset.node_traffic, axis=0) * self.daily_slots > 1)[0]
        else:
            self.traffic_data_index = np.arange(self.dataset.node_traffic.shape[1])

        # shape is [num_time_slots, station_num]
        self.traffic_data = self.dataset.node_traffic[data_range[0]:data_range[1], self.traffic_data_index].astype(
            np.float32)

        # external feature
        temporal_external_feature = []
        # weather
        if len(self.dataset.data['ExternalFeature']['Weather']) > 0:
            temporal_external_feature.append(self.dataset.data['ExternalFeature']['Weather'][data_range[0]:data_range[1]])
        # time
        if len(self.dataset.data['ExternalFeature']['Time']) > 0:
            temporal_external_feature.append(self.dataset.data['ExternalFeature']['Time'][data_range[0]:data_range[1]])
        # hot topic
        if len(self.dataset.data['ExternalFeature']['EventImpulse']) > 0:
            temporal_external_feature.append(self.dataset.data['ExternalFeature']['EventImpulse'][data_range[0]:data_range[1]])
        # concatenate
        if len(temporal_external_feature) > 0:
            temporal_external_feature = np.concatenate(temporal_external_feature, axis=-1).astype(np.float32)
            self.external_dim = temporal_external_feature.shape[1]
        else:
            self.external_dim = 0
        self.temporal_external_feature = temporal_external_feature

        # event impulse response
        event_impulse_response = self.dataset.data['ExternalFeature']['EventImpulseResponse'][data_range[0]:data_range[1]]
        self.event_impulse_response = event_impulse_response

        # spatial event influence factor
        spatial_external_feature = self.dataset.data['ExternalFeature']['EventInfluenceFactor'][data_range[0]:data_range[1]]
        self.spatial_external_feature = spatial_external_feature

        self.station_number = self.traffic_data.shape[1]

        if test_ratio > 1 or test_ratio < 0:
            raise ValueError('test_ratio ')
        self.train_test_ratio = [1 - test_ratio, test_ratio]

        # train_data shape is [num_time_slots*train_ratio, station_number], test_data shape is [num_time_slots*test_ratio, station_number]
        self.train_data, self.test_data = SplitData.split_data(self.traffic_data, self.train_test_ratio)
        # train_tef shape is [num_time_slots*train_ratio, external_dim], test_tef shape is [num_time_slots*test_ratio, external_dim]
        self.train_tef, self.test_tef = SplitData.split_data(temporal_external_feature, self.train_test_ratio)
        # train_sef shape is [num_time_slots*train_ratio, station_number], test_sef shape is [num_time_slots*test_ratio, station_number]
        self.train_sef, self.test_sef = SplitData.split_data(spatial_external_feature, self.train_test_ratio)
        # train_eir shape is [num_time_slots*train_ratio, station_number], test_eir shape is [num_time_slots*test_ratio, station_number]
        self.train_eir, self.test_eir = SplitData.split_data(event_impulse_response, self.train_test_ratio)

        # Normalize the traffic data
        if normalize:
            self.normalizer = Normalizer(self.train_data)
            self.train_data = self.normalizer.min_max_normal(self.train_data)
            self.test_data = self.normalizer.min_max_normal(self.test_data)

            self.normalizer_sef = Normalizer(self.train_sef)
            self.train_sef = self.normalizer_sef.min_max_normal(self.train_sef)
            self.test_sef = self.normalizer_sef.min_max_normal(self.test_sef)

            self.normalizer_tef = Normalizer(self.train_tef)
            self.train_tef = self.normalizer_tef.min_max_normal(self.train_tef)
            self.test_tef = self.normalizer_tef.min_max_normal(self.test_tef)

            self.normalizer_eir = Normalizer(self.train_eir)
            self.train_eir = self.normalizer_eir.min_max_normal(self.train_eir)
            self.test_eir = self.normalizer_eir.min_max_normal(self.test_eir)

        if train_data_length.lower() != 'all':
            train_day_length = int(train_data_length)
            self.train_data = self.train_data[-int(train_day_length * self.daily_slots):]
            self.train_ef = self.train_ef[-int(train_day_length * self.daily_slots):]

        # expand the test data, 把训练集中最后的输出部分的时间窗口作为测试集的输入
        expand_start_index = len(self.train_data) - \
                             max(int(self.daily_slots * self.period_len),
                                 int(self.daily_slots * 7 * self.trend_len),
                                 self.closeness_len)

        # test_data's shape is [len(self.train_data)-expand_start_index+len(self.test_data), station_num]
        self.test_data = np.vstack([self.train_data[expand_start_index:], self.test_data])
        # test_tef's shape is [len(self.train_data)-expand_start_index+len(self.test_data), external_dim]
        self.test_tef = np.vstack([self.train_tef[expand_start_index:], self.test_tef])
        # test_sef's shape is [len(self.train_data)-expand_start_index+len(self.test_data), station_num]
        self.test_sef = np.vstack([self.train_sef[expand_start_index:], self.test_sef])
        # test_eir's shape is [len(self.train_data)-expand_start_index+len(self.test_data), station_num]
        self.test_eir = np.vstack([self.train_eir[expand_start_index:], self.test_eir])

        # init move sample obj
        self.st_move_sample = ST_MoveSample(closeness_len=self.closeness_len, period_len=self.period_len,
                                            trend_len=self.trend_len, target_length=target_length,
                                            daily_slots=self.daily_slots)

        self.st_move_sample_sef = ST_MoveSample(closeness_len=self.closeness_len, period_len=self.period_len,
                                                trend_len=self.trend_len, target_length=target_length+step,
                                                daily_slots=self.daily_slots)

        # train_closeness shape is [len(train_data)-max_input_window, station_number, closeness_len, 1], train_period shape is [len(train_data)-max_input_window, station_number, period_len, 1], train_trend shape is [], train y shape is [len(train_data)-max_input_window, station_number, target_len]
        self.train_closeness, \
        self.train_period, \
        self.train_trend, \
        self.train_y = self.st_move_sample.move_sample(self.train_data)

        # test_closeness shape is [len(test_data)-max_input_window, station_number, closeness_len, 1], test_period shape is [len(test_data)-max_input_window, station_number, period_len, 1], test_trend shape is [], test y shape is [len(test_data)-max_input_window, station_number, target_len]
        self.test_closeness, \
        self.test_period, \
        self.test_trend, \
        self.test_y = self.st_move_sample.move_sample(self.test_data)

        self.train_sequence_len = max((len(self.train_closeness), len(self.train_period), len(self.train_trend)))
        self.test_sequence_len = max((len(self.test_closeness), len(self.test_period), len(self.test_trend)))

        # temporal external feature, train_tef shape is [train_sequence_len, external_dim], test_tef shape is [test_sequence_len, external_dim], 原有的这种划分方法相当于在做预测时仅使用一个时间步的额外特征施加影响
        # self.train_tef = self.train_tef[-self.train_sequence_len - target_length: -target_length]
        # self.test_tef = self.test_tef[-self.test_sequence_len - target_length: -target_length]

        # [len(test_data)-max_input_window, external_dim, closeness_len, 1]
        self.train_tef_closeness, \
        self.train_tef_period, \
        self.train_tef_trend, \
        self.train_tef_y = self.st_move_sample.move_sample(self.train_tef)

        self.test_tef_closeness, \
        self.test_tef_period, \
        self.test_tef_trend, \
        self.test_tef_y = self.st_move_sample.move_sample(self.test_tef)

        self.train_sequence_tef_len = max(len(self.train_tef_closeness), len(self.train_tef_period),
                                          len(self.train_tef_trend))
        self.test_sequence_tef_len = max(len(self.test_tef_closeness), len(self.test_tef_period),
                                         len(self.test_tef_trend))

        # spatial external feature, train_sef shape is [train_sequence_len, station_number], test_sef shape is [test_sequence_len, station_number]
        # self.train_sef = self.train_sef[-self.train_sequence_len - target_length: -target_length]
        # self.test_sef = self.test_sef[-self.test_sequence_len - target_length: -target_length]

        # [len(test_data)-max_input_window, external_dim, closeness_len, 1]
        self.train_sef_closeness, \
        self.train_sef_period, \
        self.train_sef_trend, \
        self.train_sef_y = self.st_move_sample_sef.move_sample(self.train_sef)

        self.test_sef_closeness, \
        self.test_sef_period, \
        self.test_sef_trend, \
        self.test_sef_y = self.st_move_sample_sef.move_sample(self.test_sef)

        # [len(test_data)-max_input_window, external_dim, closeness_len, 1]
        self.train_eir_closeness, \
        self.train_eir_period, \
        self.train_eir_trend, \
        self.train_eir_y = self.st_move_sample.move_sample(self.train_eir)

        self.test_eir_closeness, \
        self.test_eir_period, \
        self.test_eir_trend, \
        self.test_eir_y = self.st_move_sample.move_sample(self.test_eir)

        if with_tpe:

            # Time position embedding
            self.closeness_tpe = np.array(range(1, self.closeness_len + 1), dtype=np.float32)
            self.period_tpe = np.array(range(1 * int(self.daily_slots),
                                             self.period_len * int(self.daily_slots) + 1,
                                             int(self.daily_slots)), dtype=np.float32)
            self.trend_tpe = np.array(range(1 * int(self.daily_slots) * 7,
                                            self.trend_len * int(self.daily_slots) * 7 + 1,
                                            int(self.daily_slots) * 7), dtype=np.float32)

            self.train_closeness_tpe = np.tile(np.reshape(self.closeness_tpe, [1, 1, -1, 1]),
                                               [len(self.train_closeness), len(self.traffic_data_index), 1, 1])
            self.train_period_tpe = np.tile(np.reshape(self.period_tpe, [1, 1, -1, 1]),
                                            [len(self.train_period), len(self.traffic_data_index), 1, 1])
            self.train_trend_tpe = np.tile(np.reshape(self.trend_tpe, [1, 1, -1, 1]),
                                           [len(self.train_trend), len(self.traffic_data_index), 1, 1])

            self.test_closeness_tpe = np.tile(np.reshape(self.closeness_tpe, [1, 1, -1, 1]),
                                              [len(self.test_closeness), len(self.traffic_data_index), 1, 1])
            self.test_period_tpe = np.tile(np.reshape(self.period_tpe, [1, 1, -1, 1]),
                                           [len(self.test_period), len(self.traffic_data_index), 1, 1])
            self.test_trend_tpe = np.tile(np.reshape(self.trend_tpe, [1, 1, -1, 1]),
                                          [len(self.test_trend), len(self.traffic_data_index), 1, 1])

            self.tpe_dim = self.train_closeness_tpe.shape[-1]

            # concat temporal feature with time position embedding
            self.train_closeness = np.concatenate((self.train_closeness, self.train_closeness_tpe,), axis=-1)
            self.train_period = np.concatenate((self.train_period, self.train_period_tpe,), axis=-1)
            self.train_trend = np.concatenate((self.train_trend, self.train_trend_tpe,), axis=-1)

            self.test_closeness = np.concatenate((self.test_closeness, self.test_closeness_tpe,), axis=-1)
            self.test_period = np.concatenate((self.test_period, self.test_period_tpe,), axis=-1)
            self.test_trend = np.concatenate((self.test_trend, self.test_trend_tpe,), axis=-1)
        else:
            self.tpe_dim = None
