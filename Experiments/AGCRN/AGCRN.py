import os
import GPUtil
import torch
import argparse
import configparser

from datetime import datetime
from UCTB.model.AGCRN import AGCRN
from UCTB.utils.utils_AGCRN import Trainer
from UCTB.dataset import NodeTrafficLoader
from UCTB.utils.utils_AGCRN import get_dataloader_AGCRN
from UCTB.evaluation import metric

# Is GPU available
deviceIDs = GPUtil.getAvailable(order='last', limit=8, maxLoad=1, maxMemory=0.2,
                                includeNan=False, excludeID=[], excludeUUID=[])
if len(deviceIDs) == 0:
    current_device = '-1'
else:
    current_device = str(deviceIDs[0])
DEVICE = 'cuda:{}'.format(current_device)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--mode', default='train', type=str)
args.add_argument('--debug', default='False', type=eval)
args.add_argument('--model', default='AGCRN', type=str)
args.add_argument('--cuda', default=True, type=bool)

DATASETPATH = os.path.abspath('.')+'/params.conf'

#get configuration
config = configparser.ConfigParser()
config.read(DATASETPATH)
#device
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
#data
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default="std", type=str)
args.add_argument(
    '--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph',
                  default=config['data']['default_graph'], type=eval)
#model
args.add_argument(
    '--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument(
    '--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument(
    '--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument(
    '--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument(
    '--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument(
    '--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument(
    '--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate',
                  default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step',
                  default=config['train']['lr_decay_step'], type=str)
args.add_argument(
    '--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience',
                  default=config['train']['early_stop_patience'], type=int)
args.add_argument(
    '--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm',
                  default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'],
                  type=eval, help='use real value for loss calculation')
#test
args.add_argument(
    '--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument(
    '--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)

# data loader parameters
args.add_argument("--dataset", default='Bike', type=str,
                  help="configuration file path")
args.add_argument("--city", default='NYC', type=str)
args.add_argument("--closeness_len", default=6, type=int)
args.add_argument("--period_len", default=7, type=int)
args.add_argument("--trend_len", default=4, type=int)
args.add_argument("--data_range", default="all", type=str)
args.add_argument("--train_data_length", default="all", type=str)
args.add_argument("--test_ratio", default=0.1, type=float)
args.add_argument("--MergeIndex", default=1, type=int)
args.add_argument("--MergeWay", default="sum", type=str)

args = args.parse_args()

# loading data
data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                data_range=args.data_range, train_data_length=args.train_data_length,
                                test_ratio=float(args.test_ratio),
                                closeness_len=args.closeness_len,
                                period_len=args.period_len,
                                trend_len=args.trend_len,
                                normalizer=args.normalizer,
                                MergeIndex=args.MergeIndex,
                                MergeWay=args.MergeWay)

args.num_nodes = data_loader.station_number

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader_AGCRN(data_loader,
                                                                     tod=args.tod, batchsize=args.batch_size, dow=False,
                                                                     weather=False, single=False)


#model build
model = AGCRN(args)
model = model.to(args.device)


#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, 'model_dir', "{}_{}_{}_{}_{}_{}".format(
    args.dataset, args.city, args.MergeIndex, args.closeness_len, args.period_len, args.trend_len))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print("log_dir:", log_dir)
args.log_dir = log_dir

#Train Or Test
trainer = Trainer(model, train_loader, val_loader, test_loader, scaler, args)
if args.mode == 'train':
    # Train
    trainer.train()

model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))

print("Load saved model")

# Test
test_prediction = trainer.test(
    model, trainer.args, test_loader, scaler, trainer.logger)

test_rmse = metric.rmse(prediction=test_prediction.squeeze(
), target=data_loader.test_y.squeeze(), threshold=0)

print('Test RMSE', test_rmse)
