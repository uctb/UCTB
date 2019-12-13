import sys
sys.path.append('../')
import numpy as np
from local_path import *


class grid_config(object):
    def __init__(self, grid_width, grid_height, time_range, time_fitness):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.time_fitness = time_fitness
        self.time_range = time_range

    def parse_location(self, grid_lat_lng_str):

        grid_lat_lng = [e.strip(' ') for e in grid_lat_lng_str.split('\n') if len(e.strip(' ')) > 0]

        grid_lat_lng = [[e1 for e1 in e.replace('\t', ' ').split(' ') if len(e1) > 0][-2:] for e in grid_lat_lng]

        self.grid_lat_lng = [[float(e[0]), float(e[1])] for e in grid_lat_lng]

    def get_lat_lng_index(self, lat, lng):

        return -1 if lat < self.grid_lat_lng[0][0] else np.where(np.array(self.grid_lat_lng)[:, 0] <= lat)[0][-1],\
               -1 if lng < self.grid_lat_lng[0][1] else np.where(np.array(self.grid_lat_lng)[:, 1] <= lng)[0][-1]


class DiDi_Xian_DataConfig(object):
    def __init__(self):
        self.time_range = ['2016-10-01', '2016-11-30']
        self.time_fitness = 60
        self.grid_width = 16
        self.grid_height = 16

        self.data_path = os.path.join(didi_data_path, 'xian')
        self.raw_data_path = self.data_path

        self.file_list = [e for e in os.listdir(self.raw_data_path) if e.endswith('.json') and 'gps' in e.lower()]
        self.col_index = {
            'start_station_id': 3,
            'start_time': 1,
            'end_station_id': 5,
            'end_time': 2
        }

        self.file_col_index = {
            'time': 1,
            'lat': 3,
            'lng': 2
        }

        self.grid_lat_lng = """
        
            0	34.20829427	108.91118
            1	34.21279088	108.9166219
            2	34.21728749	108.9220638
            3	34.2217841	108.9275057
            4	34.2262807	108.9329476
            5	34.23077731	108.9383895
            6	34.23527392	108.9438314
            7	34.23977053	108.9492734
            8	34.24426714	108.9547153
            9	34.24876374	108.9601572
            10	34.25326035	108.9655991
            11	34.25775696	108.971041
            12	34.26225357	108.9764829
            13	34.26675018	108.9819248
            14	34.27124678	108.9873667
            15	34.27574339	108.9928086
            16	34.28024	108.9982505


            """

        self.grid_config = grid_config(grid_height=self.grid_height, grid_width=self.grid_width,
                                       time_range=self.time_range, time_fitness=self.time_fitness)
        self.grid_config.parse_location(self.grid_lat_lng)

        # key: station_id, value: [build_time, lat, lng, name]
        self.stations = {}
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                index = i*self.grid_width + j
                lat = (self.grid_config.grid_lat_lng[i][0] + self.grid_config.grid_lat_lng[i+1][0]) / 2
                lng = (self.grid_config.grid_lat_lng[j][1] + self.grid_config.grid_lat_lng[j+1][1]) / 2
                self.stations[str(index)] = [i*self.grid_width + j,
                                                             lat, lng, 'grid_%s' % index]

        self.stations_ordered = [e[0] for e in sorted(self.stations.items(), key=lambda x: int(x[1][0]), reverse=False)]


class DiDi_Chengdu_DataConfig(object):
    def __init__(self):
        self.time_range = ['2016-10-01', '2016-11-30']
        self.time_fitness = 60
        self.grid_width = 16
        self.grid_height = 16

        self.data_path = os.path.join(didi_data_path, 'chengdu')
        self.raw_data_path = self.data_path

        self.file_list = [e for e in os.listdir(self.raw_data_path) if e.endswith('.json') and 'gps' in e.lower()]
        self.col_index = {
            'start_station_id': 3,
            'start_time': 1,
            'end_station_id': 5,
            'end_time': 2
        }

        self.file_col_index = {
            'time': 1,
            'lat': 3,
            'lng': 2
        }

        self.grid_lat_lng = """

            0	30.65580427	104.04214
            1	30.66030088	104.047371
            2	30.66479749	104.052602
            3	30.6692941	104.0578331
            4	30.6737907	104.0630641
            5	30.67828731	104.0682951
            6	30.68278392	104.0735261
            7	30.68728053	104.0787571
            8	30.69177714	104.0839881
            9	30.69627374	104.0892192
            10	30.70077035	104.0944502
            11	30.70526696	104.0996812
            12	30.70976357	104.1049122
            13	30.71426018	104.1101432
            14	30.71875678	104.1153742
            15	30.72325339	104.1206053
            16	30.72775	104.1258363

            """

        self.grid_config = grid_config(grid_height=self.grid_height, grid_width=self.grid_width,
                                       time_range=self.time_range, time_fitness=self.time_fitness)
        self.grid_config.parse_location(self.grid_lat_lng)

        # key: station_id, value: [build_time, lat, lng, name]
        self.stations = {}
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                index = i * self.grid_width + j
                lat = (self.grid_config.grid_lat_lng[i][0] + self.grid_config.grid_lat_lng[i + 1][0]) / 2
                lng = (self.grid_config.grid_lat_lng[j][1] + self.grid_config.grid_lat_lng[j + 1][1]) / 2
                self.stations[str(index)] = [i * self.grid_width + j,
                                             lat, lng, 'grid_%s' % index]

        self.stations_ordered = [e[0] for e in sorted(self.stations.items(), key=lambda x: int(x[1][0]), reverse=False)]