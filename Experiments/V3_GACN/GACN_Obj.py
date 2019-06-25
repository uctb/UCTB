import os

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import GACN
from UCTB.evaluation import metric

from Experiments.utils import model_dir_path


def gacn_param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Metro')
    parser.add_argument('--City', default='ShanghaiV1')
    # network parameter
    parser.add_argument('--T', default='6', type=int)
    parser.add_argument('--K', default='0')
    parser.add_argument('--L', default='1')
    parser.add_argument('--Graph', default='Correlation')
    parser.add_argument('--GALLayers', default='4', type=int)
    parser.add_argument('--GALUnits', default='32', type=int)
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
    parser.add_argument('--lr', default='1e-4', type=float)
    parser.add_argument('--ESlength', default='500', type=int)
    parser.add_argument('--patience', default='0.1', type=float)
    parser.add_argument('--BatchSize', default='64', type=int)
    # device parameter
    parser.add_argument('--Device', default='0', type=str)
    # version control
    parser.add_argument('--Group', default='Debug')
    parser.add_argument('--CodeVersion', default='Shanghai_GACN2')
    return parser


parser = gacn_param_parser()
args = parser.parse_args()

model_dir = os.path.join(model_dir_path, args.Group)

code_version = 'GACN_{}_K{}L{}_{}'.format(''.join([e[0] for e in args.Graph.split('-')]),
                                          args.K, args.L, args.CodeVersion)

# Config data loader
data_loader = NodeTrafficLoader(dataset=args.Dataset, city=args.City,
                                data_range=args.DataRange, train_data_length=args.TrainDays, test_ratio=0.1,
                                normalize=True,
                                T=args.T, TI=args.TI, TD=args.TD, TC=args.TC, graph=args.Graph, with_lm=True)

de_normalizer = data_loader.normalizer.min_max_denormal

GACN_Obj = GACN(num_node=data_loader.station_number,
                input_dim=1,
                time_embedding_dim=data_loader.tpe_position_index.shape[-1],
                external_feature_dim=data_loader.external_dim,
                T=int(args.T),
                gcl_k=int(args.K),
                gcl_layers=int(args.L),
                gal_layers=int(args.GALLayers),
                gal_units=int(args.GALUnits),
                gal_num_heads=int(args.GALHeads),
                dense_units=int(args.DenseUnits),
                lr=float(args.lr),
                code_version=code_version,
                model_dir=model_dir,
                GPU_DEVICE=args.Device)

GACN_Obj.build()

# # Training
if args.Train == 'True':
    GACN_Obj.fit(input=data_loader.train_x,
                 laplace_matrix=data_loader.LM[0],
                 target=data_loader.train_y,
                 time_embedding=data_loader.tpe_position_index,
                 external_input=data_loader.train_ef,
                 batch_size=int(args.BatchSize),
                 max_epoch=int(args.Epoch),
                 early_stop_method='t-test',
                 early_stop_length=int(args.ESlength),
                 early_stop_patience=float(args.patience))

GACN_Obj.load(code_version)

# Evaluate
test_rmse = GACN_Obj.evaluate(input=data_loader.test_x,
                              laplace_matrix=data_loader.LM[0],
                              target=data_loader.test_y,
                              time_embedding=data_loader.tpe_position_index,
                              external_input=data_loader.test_ef,
                              cache_volume=4,
                              metrics=[metric.rmse],
                              de_normalizer=de_normalizer,
                              threshold=0)

print('Test result', test_rmse)