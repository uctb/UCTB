import numpy as np

from local_path import *
from dateutil.parser import parse
from Utils.json_api import getJsonDataFromPath


class grid_config(object):
    def __init__(self, grid_width, grid_height, time_range, time_fitness):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.time_fitness = time_fitness
        self.time_range = time_range

    def parse_location(self, grid_lat_lng_str):

        grid_lat_lng = [e.strip(' ') for e in grid_lat_lng_str.split('\n') if len(e.strip(' ')) > 0]

        grid_lat_lng = [[e1 for e1 in e.split('\t') if len(e1) > 0][-2:] for e in grid_lat_lng]

        self.grid_lat_lng = [[float(e[0]), float(e[1])] for e in grid_lat_lng]

    def get_lat_lng_index(self, lat, lng):

        return -1 if lat < self.grid_lat_lng[0][0] else np.where(np.array(self.grid_lat_lng)[:, 0] <= lat)[0][-1],\
               -1 if lng < self.grid_lat_lng[0][1] else np.where(np.array(self.grid_lat_lng)[:, 1] <= lng)[0][-1]



class Bike_NYC_DataConfig(object):
    def __init__(self):
        self.time_range = ['2013-07-01', '2017-09-30']
        self.time_fitness = 60
        self.grid_width = 20
        self.grid_height = 20

        self.grid_lat_lng = """

                0	40.6404988		-74.1244422
                1	40.6494921		-74.1125582
                2	40.6584854		-74.1006742
                3	40.6674787		-74.0887902
                4	40.676472		-74.0769062
                5	40.6854653		-74.0650222
                6	40.6944586		-74.0531382
                7	40.7034519		-74.0412542
                8	40.7124452		-74.0293702
                9	40.7214385		-74.0174862
                10	40.7304318		-74.0056022
                11	40.7394251		-73.9937182
                12	40.7484184		-73.9818342
                13	40.7574117		-73.9699502
                14	40.766405		-73.9580662
                15	40.7753983		-73.9461822
                16	40.7843916		-73.9342982
                17	40.7933849		-73.9224142
                18	40.8023782		-73.9105302
                19	40.8113715		-73.8986462
                20	40.8203648		-73.8867622

                """

        self.grid_config = grid_config(grid_height=self.grid_height, grid_width=self.grid_width,
                                       time_range=self.time_range, time_fitness=self.time_fitness)

        self.grid_config.parse_location(self.grid_lat_lng)

        # key: station_id, value: [build_time, lat, lng, name]
        self.stations = getJsonDataFromPath(os.path.join(data_dir, 'Bike_NYC_Stations.json'))

        self.stations_ordered = [e[0] for e in
                                 sorted(self.stations.items(), key=lambda x: parse(x[1][0]), reverse=False)]


class Bike_Chicago_DataConfig(object):
    def __init__(self):
        self.time_range = ['2013-07-01', '2017-09-30']
        self.time_fitness = 60
        self.grid_width = 20
        self.grid_height = 20

        self.grid_lat_lng = """

                0	41.8105465		-87.79764863
                1	41.8195398		-87.78554863
                2	41.8285331		-87.77344863
                3	41.8375264		-87.76134863
                4	41.8465197		-87.74924863
                5	41.855513		-87.73714863
                6	41.8645063		-87.72504863
                7	41.8734996		-87.71294863
                8	41.8824929		-87.70084863
                9	41.8914862		-87.68874863
                10	41.9004795		-87.67664863
                11	41.9094728		-87.66454863
                12	41.9184661		-87.65244863
                13	41.9274594		-87.64034863
                14	41.9364527		-87.62824863
                15	41.945446		-87.61614863
                16	41.9544393		-87.60404863
                17	41.9634326		-87.59194863
                18	41.9724259		-87.57984863
                19	41.9814192		-87.56774863
                20	41.9904125		-87.55564863

                """

        self.grid_config = grid_config(grid_height=self.grid_height, grid_width=self.grid_width,
                                       time_range=self.time_range, time_fitness=self.time_fitness)
        self.grid_config.parse_location(self.grid_lat_lng)

        # key: station_id, value: [build_time, lat, lng, name]
        self.stations = getJsonDataFromPath(os.path.join(data_dir, 'Chicago_Stations.json'))

        self.stations_ordered = [e[0] for e in
                                 sorted(self.stations.items(), key=lambda x: parse(x[1][0]), reverse=False)]


class Bike_DC_DataConfig(object):
    def __init__(self):
        self.time_range = ['2013-07-01', '2017-09-30']
        self.time_fitness = 60
        self.grid_width = 20
        self.grid_height = 20

        self.grid_lat_lng = """
                
                0	38.81582	-77.14466
                1	38.82481305	-77.13310515
                2	38.8338061	-77.1215503
                3	38.84279915	-77.10999545
                4	38.8517922	-77.0984406
                5	38.86078525	-77.08688575
                6	38.8697783	-77.0753309
                7	38.87877135	-77.06377605
                8	38.8877644	-77.0522212
                9	38.89675745	-77.04066635
                10	38.9057505	-77.0291115
                11	38.91474355	-77.01755665
                12	38.9237366	-77.0060018
                13	38.93272965	-76.99444695
                14	38.9417227	-76.9828921
                15	38.95071575	-76.97133725
                16	38.9597088	-76.9597824
                17	38.96870185	-76.94822755
                18	38.9776949	-76.9366727
                19	38.98668795	-76.92511785
                20	38.995681	-76.913563

                """

        self.grid_config = grid_config(grid_height=self.grid_height, grid_width=self.grid_width,
                                       time_range=self.time_range, time_fitness=self.time_fitness)
        self.grid_config.parse_location(self.grid_lat_lng)

        # key: station_id, value: [build_time, lat, lng, name]
        self.stations = getJsonDataFromPath(os.path.join(data_dir, 'DC_Stations.json'))

        self.stations_ordered = [e[0] for e in
                                 sorted(self.stations.items(), key=lambda x: int(x[1][0]), reverse=False)]


class ChargeStation_Beijing_DataConfig(object):
    def __init__(self):
        self.time_range = ['2018-03-21 13:00:00', '2018-08-14 23:00:00']
        self.time_fitness = 60

        # key: station_id, value: [build_time, lat, lng, name]
        self.stations = getJsonDataFromPath(os.path.join(data_dir, 'DC_Stations.json'))

        self.stations_ordered = [e[0] for e in
                                 sorted(self.stations.items(), key=lambda x: int(x[1][0]), reverse=False)]