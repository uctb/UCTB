import os
import wget
import pickle
import tarfile
import numpy as np

class DataSet(object):
    """An object storing basic data from a formatted pickle file.

    See also `Build your own datasets <./md_file/tutorial.html>`_.

    Args:
        dataset (str): A string containing path of the dataset pickle file or a string of name of the dataset.
        city (str or ``None``): ``None`` if dataset is file path, or a string of name of the city. Default: ``None``
        data_dir (str or ``None``): The dataset directory. If set to ``None``, a directory will be created.
            If ``dataset`` is file path, ``data_dir`` should be ``None`` too. Default: ``None``

    Attributes:
        data (dict): The data directly from the pickle file. ``data`` may have a ``data['contribute_data']`` dict to
            store supplementary data.
        time_range (list): From ``data['TimeRange']`` in the format of [YYYY-MM-DD, YYYY-MM-DD] indicating the time
            range of the data.
        time_fitness (int): From ``data['TimeFitness']`` indicating how many minutes is a single time slot.
        node_traffic (np.ndarray): Data recording the main stream data of the nodes in during the time range.
            From ``data['Node']['TrafficNode']`` with shape as [time_slot_num, node_num].
        node_monthly_interaction (np.ndarray): Data recording the monthly interaction of pairs of nodes.
            Its shape is [month_num, node_num, node_num].It's from ``data['Node']['TrafficMonthlyInteraction']``
            and is used to build interaction graph.
            Its an optional attribute and can be set as an empty list if interaction graph is not needed.
        node_station_info (dict): A dict storing the coordinates of nodes. It shall be formatted as {id (may be
            arbitrary): [id (when sorted, should be consistant with index of ``node_traffic``), latitude, longitude,
            other notes]}. It's from ``data['Node']['StationInfo']`` and is used to build distance graph.
            Its an optional attribute and can be set as an empty list if distance graph is not needed.
        MergeIndex(int): A int number that used to adjust the granularity of the dataset, the granularity of the new 
            dataset is time_fitness*MergeIndex. default: 1
        MergeWay(str):can be `sum` and `average`.  default: ``sum
    """
    def __init__(self, dataset, MergeIndex, MergeWay, city=None, data_dir=None):
        self.dataset = dataset
        self.city = city
        self.MergeIndex = int(MergeIndex)
        self.MergeWay = MergeWay.lower()

        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

        if os.path.isdir(data_dir) is False:
            os.makedirs(data_dir)

        if self.city is not None:
            pkl_file_name = os.path.join(data_dir, '{}_{}.pkl'.format(self.dataset, self.city))
        else:
            pkl_file_name = self.dataset

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

        # merge data
        self.data['TimeFitness']  = int(self.data['TimeFitness']*MergeIndex)
        self.data['Node']['TrafficNode'] = self.merge_data(self.data['Node']['TrafficNode'],"node")
        self.data['Grid']['TrafficGrid'] = self.merge_data(self.data['Grid']['TrafficGrid'],"grid")
        self.data['ExternalFeature']['Weather'] = self.merge_data(self.data['ExternalFeature']['Weather'],"node")

        self.time_range = self.data['TimeRange']
        self.time_fitness = self.data['TimeFitness']

        self.node_traffic = self.data['Node']['TrafficNode']
        self.node_monthly_interaction = self.data['Node']['TrafficMonthlyInteraction']
        self.node_station_info = self.data['Node']['StationInfo']

        self.grid_traffic = self.data['Grid']['TrafficGrid']
        self.grid_lat_lng = self.data['Grid']['GridLatLng']

        self.external_feature_weather = self.data['ExternalFeature']['Weather']

    def merge_data(self,data,dataType):
        if self.MergeWay == "sum":
            func = np.sum
        elif self.MergeWay == "average":
            func = np.mean
        else:
            raise ValueError("Parameter MerWay should be sum or average")
        if data.shape[0] % self.MergeIndex is not 0:
            raise ValueError("time_slots % MergeIndex should be zero")
        
        if dataType.lower() == "node":
            new = np.zeros((data.shape[0]//self.MergeIndex,data.shape[1]),dtype=np.float32)
            for new_ind,ind in enumerate(range(0,data.shape[0],self.MergeIndex)):
                    new[new_ind,:] = func(data[ind:ind+self.MergeIndex,:],axis=0)
        elif dataType.lower() == "grid":
            new = np.zeros((data.shape[0]//self.MergeIndex,data.shape[1],data.shape[2]),dtype=np.float32)
            for new_ind,ind in enumerate(range(0,data.shape[0],self.MergeIndex)):
                    new[new_ind,:,:] = func(data[ind:ind+self.MergeIndex,:,:],axis=0)
        return new