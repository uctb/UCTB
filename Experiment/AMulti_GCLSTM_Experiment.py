import os

from local_path import tf_model_dir
from Model.AMulti_GCLSTM import AMulti_GCLSTM
from EvalClass.Accuracy import Accuracy
from DataSet.node_traffic_loader import NodeTrafficLoader

def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Bike')
    parser.add_argument('--City', default='NYC')
    # network parameter
    parser.add_argument('--T', default='6')
    parser.add_argument('--K', default='1')
    parser.add_argument('--L', default='1')
    parser.add_argument('--Graph', default='Correlation-Distance-Interaction')
    parser.add_argument('--GLL', default='1')
    parser.add_argument('--LSTMUnits', default='64')
    parser.add_argument('--GALUnits', default='64')
    parser.add_argument('--GALHeads', default='2')
    parser.add_argument('--DenseUnits', default='32')
    # Training data parameters
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='All')
    # Graph parameter
    parser.add_argument('--TC', default='0')
    parser.add_argument('--TD', default='1000')
    parser.add_argument('--TI', default='500')
    # training parameters
    parser.add_argument('--Epoch', default='5000')
    parser.add_argument('--Train', default='True')
    parser.add_argument('--lr', default='1e-3')
    parser.add_argument('--patience', default='20')
    parser.add_argument('--BatchSize', default='64')
    # device parameter
    parser.add_argument('--Device', default='1')
    # version contr0l
    parser.add_argument('--Group', default='Basic')
    parser.add_argument('--CodeVersion', default='T6')
    return parser

parser = parameter_parser()
args = parser.parse_args()

# Config data loader
data_loader = NodeTrafficLoader(args, with_lm=True)

# parse parameters
K = [int(e) for e in args.K.split(',') if len(e) > 0]
L = [int(e) for e in args.L.split(',') if len(e) > 0]

train = True if args.Train == 'True' else False

code_version = 'AMulti_GCLSTM_{}_{}_{}_K{}L{}_{}'.format(args.Dataset,
                                                         args.City,
                                                         ''.join([e[0] for e in args.Graph.split('-')]),
                                                         ''.join([str(e) for e in K]),
                                                         ''.join([str(e) for e in L]),
                                                         args.CodeVersion)

print(code_version)

AMulti_GCLSTM_Obj = AMulti_GCLSTM(num_node=data_loader.station_number,
                                  GCN_K=K,
                                  GCN_layers=L,
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim,
                                  GCLSTM_layers=int(args.GLL),
                                  gal_units=int(args.GALUnits),
                                  gal_num_heads=int(args.GALHeads),
                                  T=int(args.T),
                                  num_filter_conv1x1=int(args.DenseUnits),
                                  num_hidden_units=int(args.LSTMUnits),
                                  lr=float(args.lr),
                                  code_version=code_version,
                                  GPU_DEVICE=args.Device,
                                  model_dir=os.path.join(tf_model_dir, args.Group))

AMulti_GCLSTM_Obj.build()

# Training
if train:
    AMulti_GCLSTM_Obj.fit(input=data_loader.train_x, laplace_matrix=data_loader.LM, target=data_loader.train_y,
                          external_feature=data_loader.train_ef, batch_size=int(args.BatchSize),
                          max_epoch=int(args.Epoch),
                          early_stop_method='t-test', early_stop_patience=int(args.patience))
else:
    AMulti_GCLSTM_Obj.load(code_version)

# Evaluate
test_rmse = AMulti_GCLSTM_Obj.evaluate(input=data_loader.test_x,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       metrics=[Accuracy.RMSE], threshold=0)

print('Test result', test_rmse)