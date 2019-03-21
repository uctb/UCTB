import os
import numpy as np

from local_path import tf_model_dir
from Model.ARIMA import ARIMA
from DataSet.node_traffic_loader import gcn_data_loader
from EvalClass.Accuracy import Accuracy


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--City', default='NYC')
    # version contral
    parser.add_argument('--CodeVersion', default='V0')

    parser.add_argument('--TrainDays', default='All')
    parser.add_argument('--T', default='6')

    return parser


parser = parameter_parser()
args = parser.parse_args()

result_file = 'ARIMA_%s_%s.npy' % (args.City, args.CodeVersion)

data_loader = gcn_data_loader(args, with_lm=False)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    prediction = []

    for i in range(data_loader.station_number):

        print('*************************************************************')
        print(args.City, 'Station', i)

        try:
            model_obj = ARIMA(data_loader.train_data[:, i], [6, 0, 2])
            p = model_obj.predict(data_loader.test_x[:, :, i, 0])
        except Exception as e:
            print(e)
            p = np.zeros([data_loader.test_x[:, :, i, 0].shape[0], 1])

        prediction.append(p)

        print(np.concatenate(prediction, axis=-1).shape)

    prediction = np.concatenate(prediction, axis=-1)

    np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('RMSE', Accuracy.RMSE(prediction, data_loader.test_y, threshold=-1))