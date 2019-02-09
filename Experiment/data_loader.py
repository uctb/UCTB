import numpy as np
import datetime
import json

from local_path import *
from DataSet.utils import is_work_day
from dateutil.parser import parse
from DataPreprocess.UtilClass import MoveSample, SplitData
from Model.GraphModelLayers import GraphBuilder


def getJsonDataFromPath(fullPath, showMessage=False):
    with open(fullPath, 'r') as f:
        data = json.load(f)
    if showMessage:
        print('load', fullPath)
    return data



class gcn_data_loader(object):
    def __init__(self, args, with_lm=True):

        self.city = args.City

        traffic_data_file = '%s_Traffic.npy' % self.city
        weather_file = '%s_Weather.npy' % self.city

        time_range = ['2013-07-01', '2017-09-30']

        # traffic feature
        traffic_data = np.load(os.path.join(data_dir, traffic_data_file))
        traffic_data_index = np.where(np.mean(traffic_data, axis=0) * 24 > 1)[0]
        self.traffic_data = traffic_data[:, traffic_data_index]
        # external feature
        weather_data = np.load(os.path.join(data_dir, weather_file))
        day_list = [[1 if is_work_day(parse(time_range[1]) + datetime.timedelta(hours=e)) else 0] \
                    for e in range((parse(time_range[1]) - parse(time_range[0])).days * 24)]
        external_feature = np.concatenate((weather_data, day_list), axis=-1)

        self.station_number = self.traffic_data.shape[1]
        self.external_dim = external_feature.shape[1]

        self.train_data, self.val_data, self.test_data = SplitData.split_data(self.traffic_data, 0.8, 0.1, 0.1)
        train_ef, val_ef, test_ef = SplitData.split_data(external_feature, 0.8, 0.1, 0.1)

        closeness_move_sample = MoveSample(feature_step=1, feature_stride=1, feature_length=6, target_length=1)

        self.train_x, self.train_y = closeness_move_sample.general_move_sample(self.train_data)
        self.train_ef = train_ef[-len(self.train_x) - 1:-1]

        self.val_x, self.val_y = closeness_move_sample.general_move_sample(self.val_data)
        self.val_ef = val_ef[-len(self.val_x) - 1:-1]

        self.test_x, self.test_y = closeness_move_sample.general_move_sample(self.test_data)
        self.test_ef = test_ef[-len(self.test_x) - 1:-1]

        # reshape
        self.train_x = self.train_x.transpose([0, 2, 3, 1])
        self.val_x = self.val_x.transpose([0, 2, 3, 1])
        self.test_x = self.test_x.transpose([0, 2, 3, 1])

        self.train_y = self.train_y.reshape([-1, self.station_number])
        self.val_y = self.val_y.reshape([-1, self.station_number])
        self.test_y = self.test_y.reshape([-1, self.station_number])

        if with_lm:
            if self.city == 'DC':
                date_parser = int
            else:
                date_parser = parse
            lat_lng_list = np.array([[float(e1) for e1 in e[1][1:3]] for e in
                                     sorted(
                                         getJsonDataFromPath(os.path.join(data_dir, '%s_Stations.json' % self.city)).items(),
                                         key=lambda x: date_parser(x[1][0]), reverse=False)])

            monthly_interaction = np.load(os.path.join(data_dir, '%s_Monthly_Interaction.npy' % self.city))\
                                          [:, traffic_data_index, :][:, :, traffic_data_index]
            monthly_interaction, _, _ = SplitData.split_data(monthly_interaction, 0.8, 0.1, 0.1)
            monthly_interaction = np.sum(monthly_interaction[-12:], axis=0)
            monthly_interaction = monthly_interaction + monthly_interaction.transpose()

            lm_dict = {
                'Distance': GraphBuilder.distance_graph(lat_lng_list=lat_lng_list[traffic_data_index], threshold=1000),
                'Correlation': GraphBuilder.correlation_graph(self.train_data[-30 * 24:], threshold=0, keep_weight=False),
                'Interaction': GraphBuilder.interaction_graph(monthly_interaction, threshold=500),
            }
            self.LM = np.array([lm_dict[e] for e in args.Graph.split('-') if len(e) > 0], dtype=np.float32)



class hm_data_loader(object):
    def __init__(self, args):

        self.city = args.City

        traffic_data_file = '%s_Traffic.npy' % self.city

        time_range = ['2013-07-01', '2017-09-30']

        # traffic feature
        traffic_data = np.load(os.path.join(data_dir, traffic_data_file))
        traffic_data_index = np.where(np.mean(traffic_data, axis=0) * 24 > 1)[0]
        self.traffic_data = traffic_data[:, traffic_data_index]

        self.station_number = self.traffic_data.shape[1]

        self.train_data, self.val_data, self.test_data = SplitData.split_data(self.traffic_data, 0.8, 0.1, 0.1)

        closeness_move_sample = MoveSample(feature_step=1, feature_stride=1, feature_length=6, target_length=1)
        trend_move_sample = MoveSample(feature_step=7, feature_stride=24, feature_length=1, target_length=1)

        self.test_x, self.test_y = closeness_move_sample.general_move_sample(self.test_data)
        self.test_x_trend, self.test_y_trend = trend_move_sample.general_move_sample(
            np.concatenate(
                (np.concatenate((self.train_data, self.val_data), axis=0)[-7*24:],
                 self.test_data), axis=0))

        self.test_x_trend = self.test_x_trend[-len(self.test_x):]
        self.test_y_trend = self.test_y_trend[-len(self.test_x):]

        self.test_x = self.test_x[:, 0, :, :]
        self.test_x_trend = self.test_x_trend[:, :, 0, :]