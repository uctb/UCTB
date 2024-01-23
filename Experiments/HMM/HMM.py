import nni
import argparse

from UCTB.model import HMM
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric
from UCTB.preprocess import SplitData


parser = argparse.ArgumentParser(description="Argument Parser")
# data source
parser.add_argument('--Dataset', default='Bike')
parser.add_argument('--City', default='DC')
# network parameter
parser.add_argument('--CT', default='6', type=int)
parser.add_argument('--PT', default='7', type=int)
parser.add_argument('--TT', default='4', type=int)

parser.add_argument('--DataRange', default='All')
parser.add_argument('--TrainDays', default='365')

parser.add_argument('--num_components', type=int, default=8)
parser.add_argument('--n_iter', type=int, default=365)


args = vars(parser.parse_args())

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    args['CodeVersion'] += str(nni_sid)


data_loader = NodeTrafficLoader(dataset=args['Dataset'], city=args['City'],
                                closeness_len=args['CT'], period_len=args['PT'], trend_len=args['TT'],
                                with_lm=False, with_tpe=False, normalize=False)

model = HMM(num_components=args['num_components'], n_iter=args['n_iter'])

train_closeness, val_closeness = SplitData.split_data(data_loader.train_closeness, [0.9, 0.1])
train_period, val_period = SplitData.split_data(data_loader.train_period, [0.9, 0.1])
train_trend, val_trend = SplitData.split_data(data_loader.train_trend, [0.9, 0.1])

train_label, val_label = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

model.fit(X=(train_closeness, train_period, train_trend), y=train_label)

val_results = model.predict(X=(val_closeness, val_period, val_trend))
test_results = model.predict(X=(data_loader.test_closeness, data_loader.test_period, data_loader.test_trend))

val_rmse = metric.rmse(val_results, val_label)
test_rmse = metric.rmse(test_results, data_loader.test_y)

print(args['Dataset'], args['City'], 'val_rmse', val_rmse)
print(args['Dataset'], args['City'], 'test_rmse', test_rmse)