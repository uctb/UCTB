import os
import numpy as np

from local_path import tf_model_dir
from DataSet.node_traffic_loader import hm_data_loader
from EvalClass.Accuracy import Accuracy


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--City', default='DC')
    # version contral
    parser.add_argument('--CodeVersion', default='V0')
    return parser


parser = parameter_parser()
args = parser.parse_args()

result_file = 'HM_%s_%s.npy' % (args.City, args.CodeVersion)

d, h = 0, 6

data_loader = hm_data_loader(args)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    start_index = data_loader.traffic_data.shape[0] - data_loader.test_data.shape[0]

    prediction = []

    for i in range(data_loader.test_data.shape[0]):

        p = []

        for j in range(1, d+1):
            p.append(data_loader.traffic_data[start_index + i - j * 24])

        for k in range(1, h+1):
            p.append(data_loader.traffic_data[start_index + i - k])

        prediction.append(np.mean(p, axis=0, keepdims=True))

    prediction = np.concatenate(prediction, axis=0)

    np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('RMSE', Accuracy.RMSE(prediction, data_loader.test_data, threshold=-1))