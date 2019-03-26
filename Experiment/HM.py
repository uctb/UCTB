import os
import numpy as np

from local_path import tf_model_dir
from DataSet.node_traffic_loader import NodeTrafficLoader
from EvalClass.Accuracy import Accuracy
from Model.HM import HM


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Subway')
    parser.add_argument('--City', default='Shanghai')

    # version contral
    parser.add_argument('--d', default='7')
    parser.add_argument('--h', default='1')

    parser.add_argument('--DataRange', default='All')

    parser.add_argument('--TrainDays', default='All')

    parser.add_argument('--CodeVersion', default='V0')

    return parser


parser = parameter_parser()
args = parser.parse_args()

result_file = 'HM_%s_%s_%s.npy' % (args.Dataset, args.City, args.CodeVersion)

data_loader = NodeTrafficLoader(args, with_lm=False)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    start_index = data_loader.traffic_data.shape[0] - data_loader.test_data.shape[0]

    hm_obj = HM(d=int(args.d), h=int(args.h))

    prediction = hm_obj.predict(start_index, data_loader.traffic_data, time_fitness=data_loader.dataset.time_fitness)

    # np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('RMSE', Accuracy.RMSE(prediction, data_loader.test_data, threshold=0))