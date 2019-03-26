import os

from local_path import tf_model_dir
from Model.AMulti_GCLSTM import AMulti_GCLSTM
from EvalClass.Accuracy import Accuracy
from DataSet.node_traffic_loader import SubwayTrafficLoader, amulti_gclstm_param_parser

args = amulti_gclstm_param_parser().parse_args()

# Config data loader
data_loader = SubwayTrafficLoader(dataset=args.Dataset, city=args.City,
                                  data_range=args.DataRange, train_data_length=args.TrainDays,
                                  graph=args.Graph, TC=args.TC, TD=args.TD, TI=args.TI,
                                  with_lm=True)

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
                                  GCN_K=K if len(K)>1 else K[0],
                                  GCN_layers=L if len(L)>1 else L[0],
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
                          early_stop_method='t-test', early_stop_length=int(args.ESlength),
                          early_stop_patience=float(args.patience))
else:
    AMulti_GCLSTM_Obj.load(code_version)

# Evaluate
test_rmse = AMulti_GCLSTM_Obj.evaluate(input=data_loader.test_x,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       metrics=[Accuracy.RMSE], threshold=0)

print('Test result', test_rmse)