import os
import numpy as np

from local_path import tf_model_dir
from Experiment.data_loader import hm_data_loader
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

data_loader = hm_data_loader(args)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    prediction = np.mean(np.concatenate((data_loader.test_x, data_loader.test_x_trend), axis=1), axis=1, keepdims=True)

    np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('RMSE', Accuracy.RMSE(prediction, data_loader.test_y, threshold=0))