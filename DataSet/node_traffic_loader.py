import datetime

from DataPreprocess.time_utils import is_work_day
from DataPreprocess.preprocessor import MoveSample, SplitData
from ModelUnit.GraphModelLayers import GraphBuilder

from DataPreprocess.data_config import *
from DataSet.dataset import *


class NodeTrafficLoader(object):

    def __init__(self, args, with_lm=True):

        if args.Dataset == 'Metro':
            self.dataset = MetroDataSet(args.City)
        else:
            self.dataset = DataSet(args.Dataset, args.City)

        daily_slots = 24 * 60 / self.dataset.time_fitness

        if args.DataRange.lower() == 'all':
            data_range = [0, len(self.dataset.node_traffic)]
        else:
            data_range = [int(e) for e in args.DataRange.split(',')]
            data_range = [int(data_range[0] * daily_slots), int(data_range[1] * daily_slots)]

        num_time_slots = data_range[1] - data_range[0]

        # traffic feature
        traffic_data_index = np.where(np.mean(self.dataset.node_traffic, axis=0) * daily_slots > 1)[0]

        self.traffic_data = self.dataset.node_traffic[data_range[0]:data_range[1], traffic_data_index]
        
        # external feature
        external_feature = []
        # weather
        if hasattr(self.dataset, 'external_feature_weather') and len(self.dataset.external_feature_weather) > 0:
            external_feature.append(self.dataset.external_feature_weather[data_range[0]:data_range[1]])
        # day type
        external_feature.append(
            [[1 if is_work_day(parse(self.dataset.time_range[1])
                               + datetime.timedelta(hours=e*self.dataset.time_fitness/60)) else 0] \
             for e in range(data_range[0], num_time_slots+data_range[0])])

        external_feature = np.concatenate(external_feature, axis=-1)

        self.station_number = self.traffic_data.shape[1]
        self.external_dim = external_feature.shape[1]

        train_val_test_ratio = [float(e) for e in '0.8,0.2'.split(',')]

        self.train_data, self.test_data = SplitData.split_data(self.traffic_data, train_val_test_ratio)
        train_ef, test_ef = SplitData.split_data(external_feature, train_val_test_ratio)

        if args.TrainDays.lower() != 'all':
            train_day_length = int(args.train_day_length)
            self.train_data = self.train_data[-int(train_day_length * daily_slots):]
            train_ef = train_ef[-int(train_day_length * daily_slots):]

        target_length = 1

        move_sample = MoveSample(feature_step=1,
                                 feature_stride=1,
                                 feature_length=int(args.T),
                                 target_length=target_length)

        self.train_x, self.train_y = move_sample.general_move_sample(self.train_data)
        self.train_ef = train_ef[-len(self.train_x)-target_length: -target_length]

        self.test_x, self.test_y = move_sample.general_move_sample(self.test_data)
        self.test_ef = test_ef[-len(self.test_x)-target_length: -target_length]

        self.train_x = self.train_x.reshape([-1, int(args.T), self.station_number, 1])
        self.test_x = self.test_x.reshape([-1, int(args.T), self.station_number, 1])

        self.train_y = self.train_y.reshape([-1, self.station_number, 1])
        self.test_y = self.test_y.reshape([-1, self.station_number, 1])

        # time position embedding
        # TPE 1 : one-hot vector encoding
        self.tpe_one_hot = np.array([[1 if e1 == e else 0 for e1 in range(int(args.T))] for e in range(int(args.T))])
        # TPE 2 : position index
        self.tpe_position_index = np.array([[e] for e in range(int(args.T))])

        if with_lm:

            if args.City == 'DC' or args.City == 'Beijing':
                date_parser = int
            else:
                date_parser = parse

            self.LM = []

            for graph_name in args.Graph.split('-'):

                if graph_name.lower() == 'distance':

                    if args.Dataset == 'Metro':

                        lat_lng_list = \
                            np.array([[float(e1) for e1 in e[1][1:3]] for e in
                                      self.dataset.node_station_info.items()])

                    else:

                        lat_lng_list = \
                            np.array([[float(e1) for e1 in e[1][1:3]] for e in
                                      sorted(self.dataset.node_station_info.items(),
                                             key=lambda x: date_parser(x[1][0]), reverse=False)])

                    self.LM.append(GraphBuilder.distance_graph(lat_lng_list=lat_lng_list[traffic_data_index],
                                                               threshold=float(args.TD)))

                if graph_name.lower() == 'interaction':

                    monthly_interaction = self.dataset.node_monthly_interaction[:, traffic_data_index, :][:, :, traffic_data_index]

                    monthly_interaction, _ = SplitData.split_data(monthly_interaction, train_val_test_ratio)

                    annually_interaction = np.sum(monthly_interaction[-12:-1], axis=0)
                    annually_interaction = annually_interaction + annually_interaction.transpose()

                    self.LM.append(GraphBuilder.interaction_graph(annually_interaction, threshold=float(args.TI)))

                if graph_name.lower() == 'correlation':

                    self.LM.append(GraphBuilder.correlation_graph(self.train_data[-30 * int(daily_slots):],
                                                                  threshold=float(args.TC), keep_weight=False))

                if graph_name.lower() == 'neighbor':

                    self.LM.append(GraphBuilder.adjacent_to_lm(self.dataset.neighbor))

                if graph_name.lower() == 'line':

                    self.LM.append(GraphBuilder.adjacent_to_lm(self.dataset.line))

            self.LM = np.array(self.LM, dtype=np.float32)
