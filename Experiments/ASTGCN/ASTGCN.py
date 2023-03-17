#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import GPUtil
import argparse
import configparser
import sys
UCTBfile="/mnt/UCTB_master/"
# UCTBfile变量：填入自己系统中UCTB_master文件夹的绝对路径
sys.path.append(UCTBfile)
from UCTB.model.ASTGCN import make_model
from UCTB.model.ASTGCN import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn, normalization
from UCTB.model.ASTGCN import masked_mape_np,  masked_mae,masked_mse,masked_rmse
from UCTB.preprocess import SplitData
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.dataset import NodeTrafficLoader
from UCTB.model.ASTGCN import train_main,load_graphdata_CLY,predict_main



parser = argparse.ArgumentParser()
parser.add_argument("--config", default='/mnt/UCTB待补充模型/ASTGCN-r-pytorch_0913/configurations/PEMS04_astgcn.conf', type=str,
                    help="configuration file path")
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
args = parser.parse_args()

#config
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
batch_size = int(training_config['batch_size'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
# num_for_predict = int(data_config['num_for_predict'])
num_for_predict = 1
dataset_name = "{}_{}_{}".format(args.dataset,args.city,args.MergeIndex)
model_name = training_config['model_name']

# ctx = training_config['ctx']
# os.environ["CUDA_VISIBLE_DEVICES"] = ctx
deviceIDs = GPUtil.getAvailable(order='last', limit=8, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])
if len(deviceIDs) == 0:
    current_device = "cpu"
else:
    current_device = str(deviceIDs[0])

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:{}'.format(current_device))
print("CUDA:", USE_CUDA, DEVICE)
print("MergeIndex:", args.MergeIndex)




folder_dir = '%s_channel_%d' % (model_name, in_channels)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)


# loading data
my_data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                data_range=args.data_range, train_data_length=args.train_data_length,
                                test_ratio=float(args.test_ratio),
                                closeness_len=args.closeness_len,
                                period_len=args.period_len,
                                trend_len=args.trend_len,
                                normalize=False,
                                MergeIndex=args.MergeIndex,
                                MergeWay=args.MergeWay)


# Build Graph
graph_obj = GraphGenerator(graph='distance', data_loader=my_data_loader)


num_of_vertices = my_data_loader.station_number
len_input = my_data_loader.closeness_len + my_data_loader.period_len + my_data_loader.trend_len



#load data
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_CLY(training_config,my_data_loader, DEVICE, batch_size)
adj_mx = graph_obj.AM[0]

#build model
net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices)

#train
train_main(training_config,params_path,DEVICE,net,val_loader,train_loader,test_loader,test_target_tensor,_mean, _std,graph_signal_matrix_filename)
#
#     # predict_main(13, test_loader, test_target_tensor,metric_method, _mean, _std, 'test')

# apply the best model on the test set













