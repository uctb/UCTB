import os
import numpy as np

from local_path import tf_model_dir
from Model.ARIMA import ARIMA
from Experiment.data_loader import gcn_data_loader
from EvalClass.Accuracy import Accuracy


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--City', default='Chicago')
    # version contral
    parser.add_argument('--CodeVersion', default='V0')
    return parser


parser = parameter_parser()
args = parser.parse_args()

result_file = 'ARIMA_%s_%s.npy' % (args.City, args.CodeVersion)

data_loader = gcn_data_loader(args, with_lm=False)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    prediction = []

    for i in range(data_loader.station_number):

        print(args.City, 'Station', i)

        try:
            model_obj = ARIMA(data_loader.train_data[:, i], order=(6, 0, 1))
        except:
            model_obj = ARIMA(data_loader.train_data[:, i], order=(6, 1, 1))

        p = model_obj.predict(data_loader.test_x[:, :, i, 0])

        prediction.append(p)

    prediction = np.concatenate(prediction, axis=-1)

    np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('RMSE', Accuracy.RMSE(prediction, data_loader.test_y, threshold=0))

