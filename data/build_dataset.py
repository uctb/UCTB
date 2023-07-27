import pandas as pd
import numpy as np
import pickle

from chinese_calendar import is_workday


if __name__ == '__main__':

    TimeFitness = 60
    node_num = 58
    file_dir = '../csv/'
    TimeRange = ['2017-06-01', '2017-06-19']
    region_filename = '../csv/region_17_bike_in_hour_clean2.csv'
    traffic_filename = '17_bike_in_hour_clean2.csv'
    EventImpulse = 'EventImpulse.csv'

    # weather_filename = '../csv/Xiamen17_origin/2017_09&10_clean_xm.csv'
    EventInfluence_filename = 'EventImpulseResponse.csv'
    # POIVec_filename = file_dir+'POI_vector.txt'
    EventInfluenceFactor_filename = 'EventInfluenceFactorFitness.csv'


    # 构建结点信息
    TrafficNode = pd.read_csv(file_dir + traffic_filename, header=0, index_col=0, parse_dates=[0])
    node_id_str = TrafficNode.columns.values.tolist()
    node_id_int = [i for i in range(len(node_id_str))]
    time_slot = TrafficNode.index.values.tolist()
    time_num = len(time_slot)
    traffic = TrafficNode.values

    # station-info
    coordinate_data = pd.read_csv(file_dir + region_filename, header=0, index_col=None, sep=',',
                                  usecols=['location', 'name', 'type_0'])
                                  # skiprows=lambda x: x > 0 and x % time_num != 1)
    coordinate_data['lat'] = coordinate_data['location'].str.split(',').str[0]
    coordinate_data['lng'] = coordinate_data['location'].str.split(',').str[1]
    station_info_column = ['id', 'build-time', 'lat', 'lng', 'name', 'type']
    StationInfo = coordinate_data.reindex(columns=station_info_column)
    StationInfo['id'] = node_id_int
    StationInfo['build-time'] = [time_slot[i] for i in range(node_num)]
    StationInfo['name'] = coordinate_data['name']
    StationInfo['type'] = coordinate_data['type_0']

    StationInfo = StationInfo.values.tolist()

    # poi-info
    # file = open(POIVec_filename, 'r')
    # poi = []
    # lines = file.readlines()
    # for i in lines:
    #     tmp = []
    #     i = i.strip('\n')
    #     i = i.split(',')
    #     for num in i:
    #         if num != '':
    #             num = int(num)
    #             tmp.append(num)
    #     poi.append(tmp)
    # poi = np.array(poi)
    # print("poi shape is", poi.shape)


    # external weather
    # Weather = pd.read_csv(weather_filename, header=0, index_col=None,
    #                       usecols=['temperature', 'dew_point', 'humidity', 'wind_speed', 'pressure', 'condition_id', 'wind_id'],
    #                       nrows=time_num).values
    # print("Weather shape is:", Weather.shape)

    # external Time
    timeIndex = pd.to_datetime(TrafficNode.index)
    timeFeature = pd.DataFrame(index=timeIndex,
                               columns=['workday', 'hour', 'day_of_week', 'morning_peak', 'evening_peak'])
    for idx in timeIndex:
        timeFeature.loc[idx, 'workday'] = 1 if is_workday(idx) else 0
        timeFeature.loc[idx, 'hour'] = idx.hour
        timeFeature.loc[idx, 'day_of_week'] = idx.dayofweek
        timeFeature.loc[idx, 'morning_peak'] = 1 if idx.hour == 7 or idx.hour == 8 else 0
        timeFeature.loc[idx, 'evening_peak'] = 1 if idx.hour == 17 or idx.hour == 18 else 0
    timeFeature = timeFeature.values
    print("timeFeature shape is", timeFeature.shape)

    # External Feature
    event_impulse_data = pd.read_csv(file_dir + EventImpulse, header=0, index_col=0).values
    event_impulse_response = pd.read_csv(file_dir + EventInfluence_filename, header=0, index_col=0).values
    event_influence_factor = pd.read_csv(file_dir + EventInfluenceFactor_filename, header=0, index_col=0).values
    print("EventImpulse shape is", event_impulse_data.shape)
    print("EventImpulseResponse shape is", event_impulse_response.shape)
    print("EventInfluenceFactor shape is", event_influence_factor.shape)

    # 拼接Time和weather
    # ExternalFeature = np.concatenate((Weather, timeFeature), axis=1)

    my_dataset = {"TimeRange": TimeRange,
                  "TimeFitness": TimeFitness,
                  "Node": {"TrafficNode": traffic,
                           "TrafficMonthlyInteraction": np.array([]),
                           "StationInfo": StationInfo,
                           "POI": []
                           },
                  "Grid": {
                      "TrafficGrid": [],
                      "GridLatLng": [],
                      "POI": []
                  },
                  "ExternalFeature": {
                      "Weather": [],
                      "Time": [],
                      "EventImpulse": event_impulse_data,
                      "EventImpulseResponse": event_impulse_response,
                      "EventInfluenceFactor": event_influence_factor
                  }}

    pkl_file_name = '/Users/hyymmmint/Documents/XMU/project/uctb-nas/papers/TITS2023/data/pkl/bike_hour2' + '.pkl'
    with open(pkl_file_name, 'wb') as handle:
        pickle.dump(my_dataset, handle, protocol=3)

    print('over')
