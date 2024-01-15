import argparse
from UCTB.dataset import NodeTrafficLoader
import os
from UCTB.utils.utils_GraphWaveNet import *
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.dataset import NodeTrafficLoader, data_loader
from UCTB.evaluation import metric
from UCTB.utils.utils_MTGNN import *
from UCTB.model.MTGNN import gtnet


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')

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
# 需要改下
predefined_A = load_adj(args.adj_data)
predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
predefined_A = predefined_A.to(device)

model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
model = model.to(device)
print(args)
print('The recpetive field size is', model.receptive_field)
nParams = sum([p.nelement() for p in model.parameters()])
print('Number of model parameters is', nParams)
engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, device, args.cl)
print("start training...",flush=True)
his_loss =[]
val_time = []
train_time = []
minl = 1e5
for i in range(1,args.epochs+1):
    train_loss = []
    train_mape = []
    train_rmse = []
    t1 = time.time()
    data_dict['train_loader'].shuffle()
    for iter, (x, y) in enumerate(data_dict['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainx= trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        if iter%args.step_size2==0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes/args.num_split)
        for j in range(args.num_split):
            if j != args.num_split-1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = trainx[:, :, id, :]
            ty = trainy[:, :, id, :]
            metrics = engine.train(tx, ty[:,0,:,:],id)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
        if iter % args.print_every == 0 :
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
    t2 = time.time()
    train_time.append(t2-t1)
    #validation
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x, y) in enumerate(data_dict['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        metrics = engine.eval(testx, testy[:,0,:,:])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    s2 = time.time()
    log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    val_time.append(s2-s1)
    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
    if mvalid_loss<minl:
        torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid)  + ".pth")
        minl = mvalid_loss
print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + ".pth"))
print("Training finished")
print("The valid loss on best model is", str(round(his_loss[bestid],4)))
#valid data
outputs = []
realy = torch.Tensor(data_dict['y_val']).to(device)
realy = realy.transpose(1,3)[:,0,:,:]
for iter, (x, y) in enumerate(data_dict['val_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1,3)
    with torch.no_grad():
        preds = engine.model(testx)
        preds = preds.transpose(1,3)
    outputs.append(preds.squeeze())
yhat = torch.cat(outputs,dim=0)
yhat = yhat[:realy.size(0),...]
pred = yhat
vmae, vmape, vrmse = metric(pred,realy)
#test data
outputs = []
realy = torch.Tensor(data_dict['y_test']).to(device)
realy = realy.transpose(1, 3)[:, 0, :, :]
for iter, (x, y) in enumerate(data_dict['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    with torch.no_grad():
        preds = engine.model(testx)
        preds = preds.transpose(1, 3)
    outputs.append(preds.squeeze())
yhat = torch.cat(outputs, dim=0).cpu().numpy()
yhat = yhat[:realy.size(0), ...]
