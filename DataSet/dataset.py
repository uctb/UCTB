import pickle
from local_path import *


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