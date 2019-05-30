import os

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import AMulti_GCLSTM
from UCTB.evaluation import metric

from Experiments.utils import model_dir_path

def amulti_gclstm_param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Bike')
    parser.add_argument('--City', default='NYC')
    # network parameter
    parser.add_argument('--T', default='6', type=int)
    parser.add_argument('--K', default='1', type=int)
    parser.add_argument('--L', default='1', type=int)
    parser.add_argument('--Graph', default='Correlation')
    parser.add_argument('--GLL', default='1', type=int)
    parser.add_argument('--LSTMUnits', default='64', type=int)
    parser.add_argument('--GALUnits', default='64', type=int)
    parser.add_argument('--GALHeads', default='2', type=int)
    parser.add_argument('--DenseUnits', default='32', type=int)
    # Training data parameters
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='All')
    # Graph parameter
    parser.add_argument('--TC', default='0', type=float)
    parser.add_argument('--TD', default='1000', type=float)
    parser.add_argument('--TI', default='500', type=float)
    # training parameters
    parser.add_argument('--Epoch', default='5000', type=int)
    parser.add_argument('--Train', default='True')
    parser.add_argument('--lr', default='5e-4', type=float)
    parser.add_argument('--ESlength', default='50', type=int)
    parser.add_argument('--patience', default='0.1', type=float)
    parser.add_argument('--BatchSize', default='64', type=int)
    # device parameter
    parser.add_argument('--Device', default='0', type=str)
    # version control
    parser.add_argument('--Group', default='DebugGroup')
    parser.add_argument('--CodeVersion', default='Debug')
    return parser


parser = amulti_gclstm_param_parser()
args = parser.parse_args()

model_dir = os.path.join(model_dir_path, args.Group)
code_version = 'AMultiGCLSTM_{}_K{}L{}_{}'.format(''.join([e[0] for e in args.Graph.split('-')]),
                                                  args.K, args.L, args.CodeVersion)

# Config data loader
data_loader = NodeTrafficLoader(dataset=args.Dataset, city=args.City,
                                data_range=args.DataRange, train_data_length=args.TrainDays, test_ratio=0.1,
                                T=args.T, TI=args.TI, TD=args.TD, TC=args.TC, graph=args.Graph, with_lm=True)

AMulti_GCLSTM_Obj = AMulti_GCLSTM(num_node=data_loader.station_number,
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim,
                                  T=int(args.T),
                                  GCN_K=int(args.K),
                                  GCN_layers=int(args.L),
                                  GCLSTM_layers=int(args.GLL),
                                  gal_units=int(args.GALUnits),
                                  gal_num_heads=int(args.GALHeads),
                                  num_hidden_units=int(args.LSTMUnits),
                                  num_filter_conv1x1=int(args.DenseUnits),
                                  lr=float(args.lr),
                                  code_version=code_version,
                                  model_dir=model_dir,
                                  GPU_DEVICE=args.Device)

AMulti_GCLSTM_Obj.build()

print(args.Dataset, args.City, code_version)
print('Number of trainable variables', AMulti_GCLSTM_Obj.trainable_vars)

# # Training
if args.Train == 'True':
    AMulti_GCLSTM_Obj.fit(input=data_loader.train_x,
                          laplace_matrix=data_loader.LM,
                          target=data_loader.train_y,
                          external_feature=data_loader.train_ef,
                          early_stop_method='t-test',
                          early_stop_length=int(args.ESlength),
                          early_stop_patience=float(args.patience))

AMulti_GCLSTM_Obj.load(code_version)

# Evaluate
test_rmse = AMulti_GCLSTM_Obj.evaluate(input=data_loader.test_x,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       metrics=[metric.rmse],
                                       threshold=0)

print('Test result', test_rmse)