import os

from UCTB.dataset import NodeTrafficLoader_STMGCN
from UCTB.model import ST_MGCN
from UCTB.evaluation import metric

from Experiments.utils import model_dir_path


def amulti_gclstm_param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Bike')
    parser.add_argument('--City', default='DC')
    # network parameter
    parser.add_argument('--CT', default='3', type=int)
    parser.add_argument('--PT', default='1', type=int)
    parser.add_argument('--TT', default='0', type=int)
    parser.add_argument('--K', default='1', type=int)
    parser.add_argument('--L', default='1', type=int)
    parser.add_argument('--Graph', default='Distance')
    parser.add_argument('--LSTMUnits', default='64', type=int)
    parser.add_argument('--LSTMLayers', default='3', type=int)
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
    parser.add_argument('--CodeVersion', default='ST_MGCN_Debug')
    return parser


parser = amulti_gclstm_param_parser()
args = parser.parse_args()

model_dir = os.path.join(model_dir_path, args.Group)
code_version = 'ST_MMGCN_{}_K{}L{}_{}'.format(''.join([e[0] for e in args.Graph.split('-')]),
                                              args.K, args.L, args.CodeVersion)

# Config data loader
data_loader = NodeTrafficLoader_STMGCN(dataset=args.Dataset, city=args.City,
                                       data_range=args.DataRange, train_data_length=args.TrainDays,
                                       C_T=int(args.CT), P_T=int(args.PT), T_T=int(args.TT),
                                       TI=args.TI, TD=args.TD, TC=args.TC, graph=args.Graph, with_lm=True)

ST_MGCN_Obj = ST_MGCN(T=int(args.CT) + int(args.PT) + int(args.TT),
                      input_dim=1,
                      external_dim=data_loader.external_dim,
                      num_graph=data_loader.LM.shape[0],
                      gcl_k=args.K,
                      gcl_l=args.L,
                      lstm_units=args.LSTMUnits,
                      lstm_layers=args.LSTMLayers,
                      lr=args.lr,
                      code_version=code_version,
                      model_dir=model_dir,
                      gpu_device=args.Device)

ST_MGCN_Obj.build()

print(args.Dataset, args.City, code_version)
print(ST_MGCN_Obj.trainable_vars)

# Training
if args.Train == 'True':
    ST_MGCN_Obj.fit(traffic_flow=data_loader.train_x,
                    laplace_matrix=data_loader.LM,
                    target=data_loader.train_y,
                    external_feature=data_loader.train_ef,
                    early_stop_method='t-test',
                    early_stop_length=int(args.ESlength),
                    early_stop_patience=float(args.patience))

ST_MGCN_Obj.load(code_version)

# Evaluate
test_rmse = ST_MGCN_Obj.evaluate(traffic_flow=data_loader.test_x,
                                 laplace_matrix=data_loader.LM,
                                 target=data_loader.test_y,
                                 external_feature=data_loader.test_ef,
                                 metrics=[metric.rmse],
                                 threshold=0)

print('Test result', test_rmse)