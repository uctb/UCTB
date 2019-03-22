import pickle
from local_path import *
import numpy as np
import json


class DataSet(object):

    def __init__(self, dataset, city):

        self.dataset = dataset
        self.city = city

        with open(os.path.join(data_dir, '{}_{}.pkl'.format(self.dataset, self.city)), 'rb') as f:
            self.data = pickle.load(f)

        self.time_range = self.data['TimeRange']
        self.time_fitness = self.data['TimeFitness']

        self.node_traffic = self.data['Node']['TrafficNode']
        self.node_monthly_interaction = self.data['Node']['TrafficMonthlyInteraction']
        self.node_station_info = self.data['Node']['StationInfo']

        self.grid_traffic = self.data['Grid']['TrafficGrid']
        self.grid_lat_lng = self.data['Grid']['GridLatLng']

        self.external_feature_weather = self.data['ExternalFeature']['Weather']


class MetroDataSet(object):

    def __init__(self, city):

        if city == 'cq':
            self.time_range = ['2016-08-01', '2017-07-31']

        if city == 'sh':
            self.time_range = ['2016-07-01', '2016-09-30']

        self.time_fitness = 80
        self.node_traffic = np.load("Data/%s_Traffic.npy" % city)
        self.node_station_info = json.load(open("Data/%s_Stations.json" % city))
        self.neighbor = np.load("Data/%s_Neighbors.npy" % city)
        self.line = np.load("Data/%s_Lines.npy" % city)



