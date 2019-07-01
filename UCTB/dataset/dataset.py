import os
import wget
import pickle
import tarfile


class DataSet(object):

    def __init__(self, dataset, city, data_dir=None):

        self.dataset = dataset
        self.city = city

        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

        if os.path.isdir(data_dir) is False:
            os.makedirs(data_dir)

        pkl_file_name = os.path.join(data_dir, '{}_{}.pkl'.format(self.dataset, self.city))

        if os.path.isfile(pkl_file_name) is False:
            try:
                tar_file_name = os.path.join(data_dir, '{}_{}.tar.gz'.format(self.dataset, self.city))
                if os.path.isfile(tar_file_name) is False:
                    print('Downloading data into', data_dir)
                    wget.download('https://github.com/Di-Chai/UCTB_Data/blob/master/%s_%s.tar.gz?raw=true' %
                                  (dataset, city), tar_file_name)
                    print('Download succeed')
                else:
                    print('Found', tar_file_name)
                tar = tarfile.open(tar_file_name, "r:gz")
                file_names = tar.getnames()
                for file_name in file_names:
                    tar.extract(file_name, data_dir)
                tar.close()
                os.remove(tar_file_name)
            except Exception as e:
                print(e)
                raise FileExistsError('Download Failed')

        with open(pkl_file_name, 'rb') as f:
            self.data = pickle.load(f)

        self.time_range = self.data['TimeRange']
        self.time_fitness = self.data['TimeFitness']

        self.node_traffic = self.data['Node']['TrafficNode']
        self.node_monthly_interaction = self.data['Node']['TrafficMonthlyInteraction']
        self.node_station_info = self.data['Node']['StationInfo']

        self.grid_traffic = self.data['Grid']['TrafficGrid']
        self.grid_lat_lng = self.data['Grid']['GridLatLng']

        self.external_feature_weather = self.data['ExternalFeature']['Weather']