import os
import numpy as np
import argparse

from UCTB.model import ARIMA
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric
from UCTB.utils import multiple_process


parser = argparse.ArgumentParser(description="Argument Parser")
# data source
parser.add_argument('--Dataset', default='Bike')
parser.add_argument('--City', default='NYC')
# network parameter
parser.add_argument('--CT', default='24', type=int)

parser.add_argument('--ar', default='6', type=int)
parser.add_argument('--d', default='0', type=int)
parser.add_argument('--ma', default='1', type=int)

parser.add_argument('--sar', default='0', type=int)
parser.add_argument('--sd', default='0', type=int)
parser.add_argument('--sma', default='0', type=int)
parser.add_argument('--sp', default='0', type=int)

parser.add_argument('--DataRange', default='All')
parser.add_argument('--TrainDays', default='365')

args = vars(parser.parse_args())

data_loader = NodeTrafficLoader(dataset=args['Dataset'], city=args['City'],
                                closeness_len=int(args['CT']), period_len=0, trend_len=0,
                                data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                with_lm=False, with_tpe=False, normalize=False)


def task(share_queue, locker, data, parameters):

    print('Child process %s with pid %s' % (parameters[0], os.getpid()))

    val_collector = {}
    test_collector = {}

    for i in data:

        print('Child process %s' % (parameters[0]),
              args['Dataset'], args['City'], 'Station', i, 'total', data_loader.station_number)

        try:
            model_obj = ARIMA(time_sequence=data_loader.train_closeness[:, i, -1, 0],
                              order=[args['ar'], args['d'], args['ma']],
                              seasonal_order=[args['sar'], args['sd'], args['sma'], args['sp']])

            test_prediction = model_obj.predict(time_sequences=data_loader.test_closeness[:, i, :, 0], forecast_step=1)

            del model_obj

        except Exception as e:
            print('Converge failed with error', e)
            print('Using last as prediction')

            test_prediction = data_loader.test_closeness[:, i, -1:, :]

        test_collector[i] = test_prediction

        print('Station', i, metric.rmse(test_prediction, data_loader.test_y[:, i:i + 1], threshold=0))

    locker.acquire()
    share_queue.put([val_collector, test_collector])
    locker.release()


def reduce_fn(a, b):
    a[0].update(b[0])
    a[1].update(b[1])
    return a


if __name__ == '__main__':

    n_job = 8

    result = multiple_process(distribute_list=range(data_loader.station_number),
                              partition_func=lambda data, i, n_job:
                              [data[e] for e in range(len(data)) if e % n_job == i],
                              task_func=task, n_jobs=n_job, reduce_func=reduce_fn, parameters=[])

    test_rmse_collector = [e[1] for e in sorted(result[1].items(), key=lambda x: x[0])]

    test_rmse_collector = np.concatenate(test_rmse_collector, axis=-2)

    test_rmse = metric.rmse(test_rmse_collector, data_loader.test_y, threshold=0)

    print(args['Dataset'], args['City'], 'test_rmse', test_rmse)