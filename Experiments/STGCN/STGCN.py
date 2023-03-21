import os
from os.path import join as pjoin
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from UCTB.utils.utils_STGCN import model_test,model_train,data_gen_cly
from  UCTB.model.STGCN import cheb_poly_approx
from UCTB.evaluation.metric import scaled_laplacian
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

#args config
parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--save', type=int, default=5)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='sep')


# data loader parameters
parser.add_argument("--dataset", default='Bike', type=str,help="configuration file path")
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
args.n_his = args.closeness_len -1
n_his, n_pred = args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt

# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]



# data loading
dataset_obj, graph_obj = data_gen_cly(args)
n = graph_obj.AM.shape[-1]
args.n_route = n

print(f'>> Loading dataset with Mean: {dataset_obj.mean:.2f}, STD: {dataset_obj.std:.2f}')
W = graph_obj.AM[0]

model_name = os.path.join(os.path.abspath("./EXP/"),"{}_{}_{}".format(args.dataset,args.city,args.MergeIndex))


# Calculate graph kernel
L = scaled_laplacian(W) # L with shape [n_route, n_route]
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))


#Train
model_train(dataset_obj, blocks, args, model_name)

#Test
model_test(dataset_obj, dataset_obj.get_len('test'), n_his, n_pred, args.inf_mode, load_path=model_name)
