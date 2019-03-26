import os
import numpy as np
import xgboost as xgb

from local_path import tf_model_dir
from DataSet.node_traffic_loader import NodeTrafficLoader
from EvalClass.Accuracy import Accuracy


def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Bike')
    parser.add_argument('--City', default='NYC')
    # version contral
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='All')
    parser.add_argument('--T', default='6')
    parser.add_argument('--CodeVersion', default='V0')
    return parser


parser = parameter_parser()
args = parser.parse_args()

result_file = 'XGBoost_%s_%s.npy' % (args.City, args.CodeVersion)

data_loader = NodeTrafficLoader(args, with_lm=False)

if os.path.isfile(os.path.join(tf_model_dir, result_file)) is False:

    prediction = []

    for i in range(data_loader.station_number):

        print('*************************************************************')
        print(args.City, 'Station', i)

        train_data = xgb.DMatrix(data_loader.train_x[:, :, i, 0], label=data_loader.train_y[:, i])

        test_data = xgb.DMatrix(data_loader.test_x[:, :, i, 0], label=data_loader.test_y[:, i])

        watchlist = [(train_data, 'train'), (test_data, 'test')]

        param = {'max_depth': 5, 'verbosity ': 0, 'objective': 'reg:linear', 'eval_metric': 'rmse'}

        bst = xgb.train(param, train_data, 5, watchlist)

        p = bst.predict(test_data).reshape([-1, 1])

        prediction.append(p)

    prediction = np.concatenate(prediction, axis=-1)

    np.save(os.path.join(tf_model_dir, result_file), prediction)

else:

    prediction = np.load(os.path.join(tf_model_dir, result_file))

print('XGBoost', args.City, 'RMSE', Accuracy.RMSE(prediction, data_loader.test_y, threshold=-1))