import torch
import argparse
import os
import lightning
import numpy as np

from UCTB.model.GraphWaveNet import GraphWaveNet
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.dataset import NodeTrafficLoader
from UCTB.evaluation import metric
from lightning.pytorch.callbacks import ModelCheckpoint
from UCTB.model_unit.BaseModel import NaiveEarlyStopping, CheckpointManager
from DatasetUtils import load_dataset


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

# loading node traffic data
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

# dataset adapter
device = torch.device(args.device)
train_loader, val_loader, test_loader = load_dataset(uctb_data_loader, args.batch_size, args.batch_size, args.batch_size, device)

# build graph
graph_obj = GraphGenerator(graph='distance', data_loader=uctb_data_loader)
supports = [torch.tensor(graph_obj.AM[i], dtype=torch.float32).to(device) for i in range(len(graph_obj.AM))]

print(args)

if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]
if args.aptonly:
    supports = None


# define model
model = GraphWaveNet(
        args.device,
        args.num_nodes,
        args.dropout,
        supports=supports,
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,
        in_dim=args.in_dim,
        out_dim=args.seq_length,
        residual_channels=args.nhid,
        dilation_channels=args.nhid,
        skip_channels=args.nhid * 8,
        end_channels=args.nhid * 16,
        learning_rate=args.learning_rate
    )

# load model checkpoint
checkpoint_manager = CheckpointManager(args.save)


# define early stopping
early_stopping = NaiveEarlyStopping(
    log_dir=args.save,
    monitor='val_loss',
    patience=3,  # naive strategy
    verbose=True,
    mode='min',  # minimize val_loss
)

# save model checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    dirpath=args.save,  
    filename='GWN-{epoch:0d}-{val_loss:.5f}',  # saved file name
    save_top_k=2,  
    mode='min',  # record minimized val_loss
)

# define trainer
trainer = lightning.Trainer(
    max_epochs=args.epochs,
    log_every_n_steps=1,
    devices=1,
    accelerator="auto",
    gradient_clip_val=None,
    callbacks=[checkpoint_callback, early_stopping],
)

# training
if not checkpoint_manager.converged:
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=checkpoint_manager.best_checkpoint_path)
    checkpoint_manager.best_checkpoint_path = checkpoint_callback.best_model_path

# test
checkpoint_manager.load_parameters_from_checkpoint(model, checkpoint_manager.best_checkpoint_path)
     
predictions = trainer.predict(model=model, dataloaders=test_loader )
test_prediction = np.array(torch.cat([x for x in predictions]).detach().cpu().numpy(),dtype=np.float32)
test_prediction = uctb_data_loader.normalizer.inverse_transform(test_prediction).squeeze()

y_true = uctb_data_loader.normalizer.inverse_transform(uctb_data_loader.test_y).squeeze()
rmse_result = metric.rmse(prediction=test_prediction,target=y_true)
print("Test RMSE:", rmse_result)
