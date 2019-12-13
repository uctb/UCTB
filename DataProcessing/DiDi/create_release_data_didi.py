import pickle
import argparse

from data_config import *

params = argparse.ArgumentParser()
params.add_argument('--data', default='DiDi', type=str)
params.add_argument('--city', default='Xian', type=str)

args = params.parse_args()

"""
{
    "TimeRange": [],
    "TimeFitness": [],
    "Node": {
        "TrafficNode": [],
        "TrafficMonthlyInteraction": [],
        "StationInfo": [],
        "POI": []
    },
    "Grid": {
        "TrafficGrid": [],
        "GridLatLng": [],
        "POI": []
    },
    "ExternalFeature": {
         "Weather": []
    }
}
"""

dataset = args.data

city = args.city

data_config = eval('%s_%s_DataConfig()' % (dataset, city))


release_data = {

    "TimeRange": data_config.time_range,

    "TimeFitness": data_config.time_fitness,

    "Node": {
        "TrafficNode": np.load(os.path.join(didi_data_path, '{}_{}_Traffic_Grid.npy'.format(dataset, city))).
                               reshape([-1, data_config.grid_width*data_config.grid_height]),
        "TrafficMonthlyInteraction": np.load(os.path.join(didi_data_path, '{}_{}_Monthly_Interaction.npy'.format(dataset, city))),
        "StationInfo": [[e] + data_config.stations[e] for e in data_config.stations_ordered],
        "POI": []
    },

    "Grid": {
        "TrafficGrid": np.load(os.path.join(didi_data_path, '{}_{}_Traffic_Grid.npy'.format(dataset, city))),
        "GridLatLng": data_config.grid_config.grid_lat_lng,
        "POI": []
    },

    "ExternalFeature": {
         "Weather": []
    }
}

with open(os.path.join(release_data_dir, '{}_{}.pkl'.format(dataset, city)), 'wb') as f:
    pickle.dump(release_data, f)