import torch
import argparse
import time
import os

from UCTB.utils.utils_GraphWaveNet import *
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=1, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='./garage/metr', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
# data parameters
parser.add_argument("--dataset", default='DiDi', type=str, help="configuration file path")
parser.add_argument("--city", default='Xian', type=str)
parser.add_argument("--closeness_len", default=6, type=int)
parser.add_argument("--period_len", default=7, type=int)
parser.add_argument("--trend_len", default=4, type=int)
parser.add_argument("--data_range", default="all", type=str)
parser.add_argument("--train_data_length", default="all", type=str)
parser.add_argument("--test_ratio", default=0.1, type=float)
parser.add_argument("--MergeIndex", default=1, type=int)
parser.add_argument("--MergeWay", default="sum", type=str)

args = parser.parse_args()

# loading data
uctb_data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                     data_range=args.data_range, train_data_length=args.train_data_length,
                                     test_ratio=float(args.test_ratio),
                                     closeness_len=args.closeness_len,
                                     period_len=args.period_len,
                                     trend_len=args.trend_len,
                                     normalize=False,
                                     MergeIndex=args.MergeIndex,
                                     MergeWay=args.MergeWay)

args.num_nodes = uctb_data_loader.station_number
args.in_dim = uctb_data_loader.closeness_len + uctb_data_loader.period_len + uctb_data_loader.trend_len
args.seq_length = 1
args.save = os.path.abspath('./experiment/{}_{}_{}'.format(args.dataset, args.city, args.MergeIndex))
if not os.path.exists(args.save):
    os.makedirs(args.save)

# Build Graph
graph_obj = GraphGenerator(graph='distance', data_loader=uctb_data_loader)


device = torch.device(args.device)
data_dict = load_dataset(uctb_data_loader, args.batch_size, args.batch_size, args.batch_size)

scaler = data_dict['scaler']
    
supports = [torch.tensor(graph_obj.AM[i]).to(device) for i in range(len(graph_obj.AM))]

print(args)
t1 = time.time()
if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]
if args.aptonly:
    supports = None
engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                 args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                 adjinit)

epoch_id, loss_id = Training(args, data_dict, device, engine, scaler)

print("epoch_id:", epoch_id, "loss_id:", loss_id)

test_prediction = Test(args, data_dict, device, engine, scaler, epoch_id, loss_id)

rmse_result = metric.rmse(test_prediction.squeeze(), uctb_data_loader.test_y.squeeze(), threshold=0)
print("RMSE:", rmse_result)

t2 = time.time()
print("Total time spent: {:.4f}".format(t2 - t1))


