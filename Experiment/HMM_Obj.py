import os
import numpy as np

from local_path import tf_model_dir
from Model.HMM import HMM
from Experiment.data_loader import gcn_data_loader
from EvalClass.Accuracy import Accuracy


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--City', default='NYC')
    # version contral
    parser.add_argument('--CodeVersion', default='V0')
    return parser


parser = parameter_parser()
args = parser.parse_args()

result_file = 'HMM_%s_%s.npy' % (args.City, args.CodeVersion)

data_loader = gcn_data_loader(args, with_lm=False)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    prediction = []

    for station_index in range(data_loader.station_number):

        # train the hmm model
        try:
            hmm = HMM(num_components=8, n_iter=1000)
            hmm.fit(data_loader.train_data[:, station_index:station_index+1])
            # predict
            p = []
            for time_index in range(data_loader.test_x.shape[0]):
                p.append(hmm.predict(data_loader.test_x[time_index, :, station_index, :], length=1))
        except:
            print('Error station index', station_index)
            p = [[0] for _ in range(data_loader.test_x.shape[0])]

        prediction.append(p)

    prediction = np.concatenate(prediction, axis=-1)

    np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('RMSE', Accuracy.RMSE(prediction, data_loader.test_y, threshold=-1))