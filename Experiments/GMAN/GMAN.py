import time
import argparse
import os

from UCTB.evaluation import metric
from UCTB.model.GMAN import Graph
from UCTB.dataset import NodeTrafficLoader
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.utils.utils_GMAN import *

#args config
parser = argparse.ArgumentParser()
# data loader parameters
parser.add_argument("--dataset", default='Bike', type=str)
parser.add_argument("--city", default='NYC', type=str)
parser.add_argument("--closeness_len", default=12, type=int)
parser.add_argument("--period_len", default=0, type=int)
parser.add_argument("--trend_len", default=0, type=int)
parser.add_argument("--data_range", default="all", type=str)
parser.add_argument("--train_data_length", default="all", type=str)
parser.add_argument("--test_ratio", default=0.1, type=float)
parser.add_argument("--MergeIndex", default=1, type=int)
parser.add_argument("--MergeWay", default="sum", type=str)
parser.add_argument("--threshold_correlation", default=0.7, type=float)

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

# spatial embedding parameters
parser.add_argument('--spatial_is_directed', type=bool, default=False)
parser.add_argument('--spatial_p', type=int, default=2)
parser.add_argument('--spatial_q', type=int, default=1)
parser.add_argument('--spatial_num_walks', type=int, default=100)
parser.add_argument('--spatial_walk_length', type=int, default=80)
parser.add_argument('--spatial_dimensions', type=int, default=32)
parser.add_argument('--spatial_epochs', type=int, default=1000)


args = parser.parse_args()

#config data_loader
data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                data_range=args.data_range, train_data_length=args.train_data_length,
                                test_ratio=float(args.test_ratio),
                                closeness_len=args.closeness_len,
                                period_len=args.period_len,
                                trend_len=args.trend_len,
                                normalize=False, remove=False,
                                MergeIndex=args.MergeIndex,
                                MergeWay=args.MergeWay)

args.P = args.closeness_len + args.period_len + args.trend_len

graph_obj = GraphGenerator(graph='correlation', data_loader=data_loader,
                           threshold_distance=args.threshold_correlation)

# Global variable
adj_file = os.path.abspath("./Graph_File/{}_{}_adj.txt".format(args.dataset, args.city))
SE_file = os.path.abspath("./Graph_File/{}_{}_SE.txt".format(args.dataset, args.city))
args.SE_file = SE_file

if not os.path.exists(SE_file):
    # Generate Graph embedding
    graph_to_adj_files(graph_obj.AM[0], adj_file)

    nx_G = read_graph(adj_file)
    G = Graph(nx_G, args.spatial_is_directed, args.spatial_p, args.spatial_q)
    G.preprocess_transition_probs()

    walks = G.simulate_walks(args.spatial_num_walks, args.spatial_walk_length)
    learn_embeddings(walks, args.spatial_dimensions, SE_file, args.spatial_epochs)


model_name = os.path.abspath("model_dir/{}_{}_{}/".format(args.dataset, args.city, args.MergeIndex))
if not os.path.exists(model_name):
    os.makedirs(model_name)
args.model_file = model_name
print("model_name:", args.model_file)

log_file = os.path.abspath("log/{}_{}_{}.txt".format(args.dataset, args.city, args.MergeIndex))
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))
args.log_file = log_file
print("log_file:", args.log_file)


start = time.time()
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])

# load data
log_string(log, 'loading data...')

(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
 SE,time_fitness) = load_data(args, data_loader)


log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
log_string(log, 'data loaded!')

# Train and Test
X, TE, label, is_training, saver, sess, train_op, loss, pred = build_model(
    log, time_fitness, trainX, args,SE)

train_prediction, val_prediction = Train(
    log, args, trainX, trainY, trainTE, valX, valTE, valY, X, TE, label, is_training, saver, sess, train_op, loss, pred)

test_prediction = Test(log, args, testX, testTE, X,
                       TE, is_training, sess, pred)
test_prediction = data_loader.normalizer.inverse_transform(test_prediction)
y_true = data_loader.normalizer.inverse_transform(data_loader.test_y)
test_rmse = metric.rmse(prediction=test_prediction.squeeze(),
                        target=y_true.squeeze(), threshold=0)

print("Test RMSE:", test_rmse)
