from genericpath import exists
import math
import argparse
import sys
UCTBfile="/mnt/UCTB_master3/"
# UCTBfile变量：填入自己系统中UCTB_master文件夹的绝对路径
sys.path.append(UCTBfile)
from UCTB.utils.GMAN.GMAN_utils import *
import time, datetime
import numpy as np
import tensorflow as tf
import os
from UCTB.model.GMAN import Graph
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import argparse
import os
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.model.GMAN import NodeTrafficLoader

#args config
parser = argparse.ArgumentParser()
# data loader parameters
parser.add_argument("--dataset", default='Bike', type=str)
parser.add_argument("--city", default='NYC', type=str)
parser.add_argument("--closeness_len", default=6, type=int)
parser.add_argument("--period_len", default=7, type=int)
parser.add_argument("--trend_len", default=4, type=int)
parser.add_argument("--data_range", default="all", type=str)
parser.add_argument("--train_data_length", default="all", type=str)
parser.add_argument("--test_ratio", default=0.1, type=float)
parser.add_argument("--MergeIndex", default=1, type=int)
parser.add_argument("--MergeWay", default="sum", type=str)
parser.add_argument("--threshold_distance", default=0.1, type=float)
parser.add_argument('--P', type=int, default=12,
                    help='history steps')
parser.add_argument('--Q', type=int, default=1,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=500,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=40,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=5,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='data/PeMS.h5',
                    help='traffic file')
parser.add_argument('--SE_file', default='data/SE(PeMS).txt',
                    help='spatial emebdding file')
parser.add_argument('--model_file', default='data/GMAN(PeMS)',
                    help='save the model to disk')
parser.add_argument('--log_file', default='data/log(PeMS)',
                    help='log file')
args = parser.parse_args()

#config data_loader
data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                data_range=args.data_range, train_data_length=args.train_data_length,
                                test_ratio=float(args.test_ratio),
                                closeness_len=args.closeness_len,
                                period_len=args.period_len,
                                trend_len=args.trend_len,
                                normalize=False,remove=False,
                                MergeIndex=args.MergeIndex,
                                MergeWay=args.MergeWay)
graph_obj = GraphGenerator(graph='distance', data_loader=data_loader, threshold_distance=args.threshold_distance)


#Global variable
is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
epochs = 1000
Adj_file = os.path.abspath("./Graph_File/{}_{}_adj.txt".format(args.dataset, args.city))
print(Adj_file)
SE_file = os.path.abspath("./Graph_File/{}_{}_SE.txt".format(args.dataset, args.city))
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


#Generate Graph embeddind

graph_to_adj_files(graph_obj.AM[0], Adj_file)
nx_G = read_graph(Adj_file)
G = Graph(nx_G, is_directed, p, q)

# import pdb;pdb.set_trace()

G.preprocess_transition_probs()

walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file,epochs)


#reset args to Train
args = parser.parse_args()
args.closeness_len=12
args.period_len=0
args.trend_len=0
args.data_range="all"
args.train_data_length="all"
args.test_ratio=0.1
args.MergeIndex=1
args.MergeWay="sum"
args.P = args.closeness_len
args.SE_file = "/mnt/{}_{}_SE.txt".format(args.dataset, args.city)
basic_dir = os.path.abspath("EXP/{}_{}_{}/".format(args.dataset, args.city, args.MergeIndex))
if not os.path.exists(basic_dir):
    os.makedirs(basic_dir)
model_name = os.path.join(basic_dir, "model")
args.model_file = model_name


#log config
print("model_name:", args.model_file)
start = time.time()
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])


# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE,
 mean, std, time_fitness) = loadData_cly(args)



log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
log_string(log, 'data loaded!')

# Train and Test
Train_then_Test(log,time_fitness,trainX,args,std,SE,mean,valX,valTE,valY,testX,testTE,testY,start,trainY,trainTE)
