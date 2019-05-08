import os
import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.preprocess import ST_MoveSample
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
    parser.add_argument('--TT', default='1', type=int)
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


class NodeTrafficLoader_STMGCN(NodeTrafficLoader):

    def __init__(self,
                 dataset,
                 city,
                 C_T,
                 P_T,
                 T_T,
                 data_range='All',
                 train_data_length='All',
                 test_ratio=0.1,
                 graph='Correlation',
                 TD=1000,
                 TC=0,
                 TI=500,
                 with_lm=True,
                 data_dir=None):

        super(NodeTrafficLoader_STMGCN, self).__init__(dataset=dataset,
                                                       city=city,
                                                       data_range=data_range,
                                                       train_data_length=train_data_length,
                                                       test_ratio=test_ratio, T=None,
                                                       graph=graph, TD=TD, TC=TC, TI=TI,
                                                       with_lm=with_lm,
                                                       data_dir=data_dir)
        target_length = 1

        # expand the test data
        self.test_data = np.vstack([self.train_data[-max(int(self.daily_slots*P_T), int(self.daily_slots*7*T_T)):],
                                    self.test_data])

        st_move_sample = ST_MoveSample(C_T=C_T, P_T=P_T, T_T=T_T, target_length=1)

        self.train_closeness,\
        self.train_period,\
        self.train_trend,\
        self.train_y = st_move_sample.move_sample(self.train_data)

        self.train_closeness = self.train_closeness.transpose([0, 3, 2, 1])
        self.train_period = self.train_period[:, :, :, -1:]
        self.train_trend = self.train_trend[:, :, :, -1:]

        self.test_closeness,\
        self.test_period,\
        self.test_trend,\
        self.test_y = st_move_sample.move_sample(self.test_data)

        self.test_closeness = self.test_closeness.transpose([0, 3, 2, 1])
        self.test_period = self.test_period[:, :, :, -1:]
        self.test_trend = self.test_trend[:, :, :, -1:]

        self.train_x = np.concatenate([self.train_closeness, self.train_period, self.train_trend], axis=1)
        self.test_x = np.concatenate([self.test_closeness, self.test_period, self.test_trend], axis=1)

        # external feature
        self.train_ef = self.train_ef[-len(self.train_closeness) - target_length: -target_length]
        self.test_ef = self.test_ef[-len(self.test_closeness) - target_length: -target_length]


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
                      GPU_DEVICE=args.Device)

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