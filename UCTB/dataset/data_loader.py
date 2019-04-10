import datetime
import numpy as np

from dateutil.parser import parse

from ..preprocess.time_utils import is_work_day, is_valid_date
from ..preprocess import MoveSample, SplitData
from ..model_unit import GraphBuilder

from .dataset import DataSet


class NodeTrafficLoader(object):

    def __init__(self,
                 dataset,
                 city,
                 data_range='All',
                 train_data_length='All',
                 test_ratio=0.1,
                 T=6,
                 graph='Correlation',
                 TD=1000,
                 TC=0,
                 TI=500,
                 with_lm=True,
                 data_dir=None):

        self.dataset = DataSet(dataset, city, data_dir=data_dir)

        daily_slots = 24 * 60 / self.dataset.time_fitness

        if data_range.lower() == 'all':
            data_range = [0, len(self.dataset.node_traffic)]
        else:
            data_range = [int(e) for e in data_range.split(',')]
            data_range = [int(data_range[0] * daily_slots), int(data_range[1] * daily_slots)]

        num_time_slots = data_range[1] - data_range[0]

        # traffic feature
        traffic_data_index = np.where(np.mean(self.dataset.node_traffic, axis=0) * daily_slots > 1)[0]

        self.traffic_data = self.dataset.node_traffic[data_range[0]:data_range[1], traffic_data_index]

        # external feature
        external_feature = []
        # weather
        if len(self.dataset.external_feature_weather) > 0:
            external_feature.append(self.dataset.external_feature_weather[data_range[0]:data_range[1]])
        # day type
        external_feature.append(
            [[1 if is_work_day(parse(self.dataset.time_range[1])
                               + datetime.timedelta(hours=e * self.dataset.time_fitness / 60)) else 0] \
             for e in range(data_range[0], num_time_slots + data_range[0])])
        # Hour Feature
        hour_feature = [[(parse(self.dataset.time_range[1]) +
                         datetime.timedelta(hours=e * self.dataset.time_fitness / 60)).hour]
                        for e in range(data_range[0], num_time_slots + data_range[0])]

        external_feature.append(hour_feature)

        external_feature = np.concatenate(external_feature, axis=-1)

        self.station_number = self.traffic_data.shape[1]
        self.external_dim = external_feature.shape[1]

        if test_ratio > 1 or test_ratio < 0:
            raise ValueError('test_ratio ')
        train_test_ratio = [1 - test_ratio, test_ratio]

        self.train_data, self.test_data = SplitData.split_data(self.traffic_data, train_test_ratio)
        train_ef, test_ef = SplitData.split_data(external_feature, train_test_ratio)

        if train_data_length.lower() != 'all':
            train_day_length = int(train_data_length)
            self.train_data = self.train_data[-int(train_day_length * daily_slots):]
            train_ef = train_ef[-int(train_day_length * daily_slots):]

        if T is not None:
            target_length = 1
            move_sample = MoveSample(feature_step=1,
                                     feature_stride=1,
                                     feature_length=int(T),
                                     target_length=target_length)

            self.train_x, self.train_y = move_sample.general_move_sample(self.train_data)
            self.train_ef = train_ef[-len(self.train_x) - target_length: -target_length]

            self.test_x, self.test_y = move_sample.general_move_sample(self.test_data)
            self.test_ef = test_ef[-len(self.test_x) - target_length: -target_length]

            self.train_x = self.train_x.reshape([-1, int(T), self.station_number, 1])
            self.test_x = self.test_x.reshape([-1, int(T), self.station_number, 1])

            self.train_y = self.train_y.reshape([-1, self.station_number, 1])
            self.test_y = self.test_y.reshape([-1, self.station_number, 1])

            # time position embedding
            # TPE 1 : one-hot vector encoding
            self.tpe_one_hot = np.array([[1 if e1 == e else 0 for e1 in range(int(T))] for e in range(int(T))])
            # TPE 2 : position index
            self.tpe_position_index = np.array([[e] for e in range(int(T))])

        if with_lm:

            self.LM = []

            for graph_name in graph.split('-'):

                if graph_name.lower() == 'distance':

                    # Default order by date
                    order_parser = parse
                    try:
                        for key, value in self.dataset.node_station_info.items():
                            if is_valid_date(value[0]) is False:
                                order_parser = lambda x: x
                                print('Order by string')
                                break
                    except:
                        order_parser = lambda x: x
                        print('Order by string')

                    lat_lng_list = \
                        np.array([[float(e1) for e1 in e[1][1:3]] for e in
                                  sorted(self.dataset.node_station_info.items(),
                                         key=lambda x: order_parser(x[1][0]), reverse=False)])

                    self.LM.append(GraphBuilder.distance_graph(lat_lng_list=lat_lng_list[traffic_data_index],
                                                               threshold=float(TD)))

                if graph_name.lower() == 'interaction':
                    monthly_interaction = self.dataset.node_monthly_interaction[:, traffic_data_index, :][:, :,
                                          traffic_data_index]

                    monthly_interaction, _ = SplitData.split_data(monthly_interaction, train_test_ratio)

                    annually_interaction = np.sum(monthly_interaction[-12:-1], axis=0)
                    annually_interaction = annually_interaction + annually_interaction.transpose()

                    self.LM.append(GraphBuilder.interaction_graph(annually_interaction, threshold=float(TI)))

                if graph_name.lower() == 'correlation':
                    self.LM.append(GraphBuilder.correlation_graph(self.train_data[-30 * int(daily_slots):],
                                                                  threshold=float(TC), keep_weight=False))

            self.LM = np.array(self.LM, dtype=np.float32)


class SubwayTrafficLoader(NodeTrafficLoader):

    def __init__(self,
                 dataset,
                 city,
                 data_range='All',
                 train_data_length='All',
                 test_ratio=0.1,
                 T=6,
                 graph='Correlation',
                 TD=1000,
                 TC=0,
                 TI=500,
                 with_lm=True):

        super(SubwayTrafficLoader, self).__init__(dataset=dataset,
                                                  city=city,
                                                  data_range=data_range,
                                                  train_data_length=train_data_length,
                                                  test_ratio=test_ratio,
                                                  T=T, graph=graph, TD=TD, TC=TC, TI=TI,
                                                  with_lm=with_lm)

        if with_lm:

            LM = []

            for graph_name in graph.split('-'):

                if graph_name.lower() == 'neighbor':
                    LM.append(
                        GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_neighbors')))

                if graph_name.lower() == 'line':
                    LM.append(GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_lines')))

                if graph_name.lower() == 'transfer':
                    LM.append(
                        GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_transfer')))

            if len(LM) > 0:

                if len(self.LM) == 0:
                    self.LM = np.array(LM, dtype=np.float32)
                else:
                    self.LM = np.concatenate((self.LM, LM), axis=0)