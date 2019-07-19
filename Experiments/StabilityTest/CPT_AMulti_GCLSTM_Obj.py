import os
import numpy as np

from UCTB.dataset import NodeTrafficLoader_CPT
from UCTB.model import AMulti_GCLSTM_V1
from UCTB.evaluation import metric
from UCTB.model_unit import GraphBuilder
from UCTB.preprocess import is_work_day_china

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')


def cpt_amulti_gclstm_param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='DiDi')
    parser.add_argument('--City', default='Xian')
    # network parameter
    parser.add_argument('--CT', default='6', type=int)
    parser.add_argument('--PT', default='7', type=int)
    parser.add_argument('--TT', default='4', type=int)
    parser.add_argument('--K', default='1', type=int)
    parser.add_argument('--L', default='1', type=int)
    parser.add_argument('--Graph', default='Distance-Interaction-Correlation')
    parser.add_argument('--GLL', default='1', type=int)
    parser.add_argument('--LSTMUnits', default='64', type=int)
    parser.add_argument('--GALUnits', default='64', type=int)
    parser.add_argument('--GALHeads', default='2', type=int)
    parser.add_argument('--DenseUnits', default='32', type=int)
    parser.add_argument('--Normalize', default='True', type=str)
    # Training data parameters
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='All')
    # Graph parameter
    parser.add_argument('--TC', default='0.7', type=float)
    parser.add_argument('--TD', default='3000', type=float)
    parser.add_argument('--TI', default='100', type=float)
    # training parameters
    parser.add_argument('--Epoch', default='5000', type=int)
    parser.add_argument('--Train', default='True', type=str)
    parser.add_argument('--lr', default='1e-4', type=float)
    parser.add_argument('--ESlength', default='200', type=int)
    parser.add_argument('--patience', default='0.1', type=float)
    parser.add_argument('--BatchSize', default='128', type=int)
    # device parameter
    parser.add_argument('--Device', default='0,1', type=str)
    # version control
    parser.add_argument('--Group', default='Xian')
    parser.add_argument('--CodeVersion', default='ParamTuner')
    return parser


class SubwayTrafficLoader(NodeTrafficLoader_CPT):

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
                 workday_parser=is_work_day_china,
                 normalize=False,
                 with_lm=True):

        super(SubwayTrafficLoader, self).__init__(dataset=dataset,
                                                  city=city,
                                                  data_range=data_range,
                                                  train_data_length=train_data_length,
                                                  test_ratio=test_ratio,
                                                  graph=graph, TD=TD, TC=TC, TI=TI,
                                                  C_T=C_T, P_T=P_T, T_T=T_T,
                                                  workday_parser=workday_parser,
                                                  normalize=normalize,
                                                  with_lm=with_lm)
        if with_lm:
            LM = []
            for graph_name in graph.split('-'):
                if graph_name.lower() == 'neighbor':
                    LM.append(
                        GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_neighbors')))
                if graph_name.lower() == 'line':
                    LM.append(GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_lines')))
                if graph_name.lower() == 'transfer':
                    LM.append(
                        GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_transfer')))
            if len(LM) > 0:
                if len(self.LM) == 0:
                    self.LM = np.array(LM, dtype=np.float32)
                else:
                    self.LM = np.concatenate((self.LM, LM), axis=0)


parser = cpt_amulti_gclstm_param_parser()
args = vars(parser.parse_args())

model_dir = os.path.join(model_dir_path, args['Group'])
code_version = 'CPT_AMultiGCLSTM_{}_K{}L{}_{}'.format(''.join([e[0] for e in args['Graph'].split('-')]),
                                                      args['K'], args['L'], args['CodeVersion'])

# Config data loader
data_loader = SubwayTrafficLoader(dataset=args['Dataset'], city=args['City'],
                                  data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                  test_ratio=0.1,
                                  C_T=int(args['CT']), P_T=int(args['PT']), T_T=int(args['TT']),
                                  TI=args['TI'], TD=args['TD'], TC=args['TC'],
                                  normalize=True if args['Normalize'] == 'True' else False,
                                  graph=args['Graph'], with_lm=True)

de_normalizer = None if args['Normalize'] == 'False' else data_loader.normalizer.min_max_denormal

CPT_AMulti_GCLSTM_Obj = AMulti_GCLSTM_V1(num_node=data_loader.station_number,
                                         num_graph=data_loader.LM.shape[0],
                                         external_dim=data_loader.external_dim,
                                         C_T=int(args['CT']), P_T=int(args['PT']), T_T=int(args['TT']),
                                         GCN_K=int(args['K']),
                                         GCN_layers=int(args['L']),
                                         GCLSTM_layers=int(args['GLL']),
                                         gal_units=int(args['GALUnits']),
                                         gal_num_heads=int(args['GALHeads']),
                                         num_hidden_units=int(args['LSTMUnits']),
                                         num_filter_conv1x1=int(args['DenseUnits']),
                                         lr=float(args['lr']),
                                         code_version=code_version,
                                         model_dir=model_dir,
                                         GPU_DEVICE=args['Device'])

import json

with open(os.path.join(CPT_AMulti_GCLSTM_Obj._log_dir, 'params.json'), 'w') as f:
    json.dump(args, f)

CPT_AMulti_GCLSTM_Obj.build()

print(args['Dataset'], args['City'], code_version)
print('Number of trainable variables', CPT_AMulti_GCLSTM_Obj.trainable_vars)

# Training
if args['Train'] == 'True':
    CPT_AMulti_GCLSTM_Obj.fit(closeness_feature=data_loader.train_closeness,
                              period_feature=data_loader.train_period,
                              trend_feature=data_loader.train_trend,
                              laplace_matrix=data_loader.LM,
                              target=data_loader.train_y,
                              external_feature=data_loader.train_ef,
                              early_stop_method='t-test',
                              early_stop_length=int(args['ESlength']),
                              early_stop_patience=float(args['patience']),
                              batch_size=int(args['BatchSize']),
                              max_epoch=int(args['Epoch']))

CPT_AMulti_GCLSTM_Obj.load(code_version)

# Evaluate
test_error = CPT_AMulti_GCLSTM_Obj.evaluate(closeness_feature=data_loader.test_closeness,
                                            period_feature=data_loader.test_period,
                                            trend_feature=data_loader.test_trend,
                                            laplace_matrix=data_loader.LM,
                                            target=data_loader.test_y,
                                            external_feature=data_loader.test_ef,
                                            cache_volume=int(args['BatchSize']),
                                            metrics=[metric.rmse, metric.mape],
                                            de_normalizer=de_normalizer,
                                            threshold=0)

print('Test result', test_error)