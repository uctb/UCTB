import json
import argparse
import mxnet as mx
from UCTB.dataset import NodeTrafficLoader
from UCTB.model.STSGCN import *
from UCTB.utils.utils_STSGCN import *

#args
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    default='./config/PEMS03/STMeta_emb.json', help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--plot", help="plot network graph", action="store_true")
parser.add_argument("--save", type=bool, default=True, help="save model")

# data loader parameters
parser.add_argument("--dataset", default='Bike', type=str,
                    help="configuration file path")
parser.add_argument("--city", default='NYC', type=str)
parser.add_argument("--closeness_len", default=6, type=int)
parser.add_argument("--period_len", default=7, type=int)
parser.add_argument("--trend_len", default=4, type=int)
parser.add_argument("--data_range", default="all", type=str)
parser.add_argument("--train_data_length", default="all", type=str)
parser.add_argument("--test_ratio", default=0.1, type=float)
parser.add_argument("--MergeIndex", default=1, type=int)
parser.add_argument("--MergeWay", default="sum", type=str)
parser.add_argument("--normalize", default=False, type=bool)
args = parser.parse_args()
config_filename = args.config

#config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
batch_size = config['batch_size']
if isinstance(config['ctx'], list):
    ctx = [mx.gpu(i) for i in config['ctx']]
elif isinstance(config['ctx'], int):
    ctx = mx.gpu(config['ctx'])

#date_load
data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                data_range=args.data_range, train_data_length=args.train_data_length,
                                test_ratio=float(args.test_ratio),
                                closeness_len=args.closeness_len,
                                period_len=args.period_len,
                                trend_len=args.trend_len,
                                normalize=args.normalize,
                                MergeIndex=args.MergeIndex,
                                MergeWay=args.MergeWay)

#config_params
model_name, epochs, metric, mod, train_loader, val_loader, test_loader, normalizer, val_y, test_y, all_info = configData(
    args, data_loader, batch_size, config, ctx)


#Train Or Test
if args.test:
    epochs = 5
training(epochs, metric, mod, train_loader, val_loader,
         test_loader, normalizer, val_y, test_y, all_info)


#Print result
the_best = min(all_info, key=lambda x: x[2])
print('Last Epoch : {}\ntraining loss: {:.2f}\nvalidation loss: {:.2f}\n'
      'testing: MAPE: {:.2f}\n'
      'testing: RMSE: {:.2f}\n'.format(*the_best))

#model_save
if args.save:
    mod.save_checkpoint(model_name, epochs)
