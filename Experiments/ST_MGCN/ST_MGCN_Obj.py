import os
import nni
import GPUtil
import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import ST_MGCN
from UCTB.evaluation import metric


def stmeta_param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='Metro')
    parser.add_argument('--City', default='Shanghai')
    # network parameter
    parser.add_argument('--CT', default='6', type=int)
    parser.add_argument('--PT', default='7', type=int)
    parser.add_argument('--TT', default='4', type=int)
    parser.add_argument('--K', default='1', type=int)
    parser.add_argument('--L', default='1', type=int)
    parser.add_argument('--Graph', default='Distance-Correlation-Line')
    parser.add_argument('--LSTMUnits', default='128', type=int)
    parser.add_argument('--LSTMLayers', default='3', type=int)
    # Training data parameters
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='365')
    # Graph parameter
    parser.add_argument('--TC', default='0', type=float)
    parser.add_argument('--TD', default='1000', type=float)
    parser.add_argument('--TI', default='500', type=float)
    # training parameters
    parser.add_argument('--Epoch', default='20000', type=int)
    parser.add_argument('--Train', default='True')
    parser.add_argument('--lr', default='5e-4', type=float)
    parser.add_argument('--ESlength', default='50', type=int)
    parser.add_argument('--patience', default='0.1', type=float)
    parser.add_argument('--BatchSize', default='32', type=int)
    # device parameter
    parser.add_argument('--Device', default='0', type=str)
    # version control
    parser.add_argument('--CodeVersion', default='V0')
    # Merge times
    parser.add_argument('--MergeIndex', default=6, type=int)
    return parser


parser = stmeta_param_parser()
args = vars(parser.parse_args())

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    args['CodeVersion'] += str(nni_sid)

model_dir = os.path.join('model_dir', args['City'])
code_version = 'ST_MMGCN_{}_K{}L{}_{}_F{}'.format(''.join([e[0] for e in args['Graph'].split('-')]),
                                              args['K'], args['L'], args['CodeVersion'],int(args['MergeIndex'])*5)

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    if nni_params:
        current_device = str(deviceIDs[int(nni_sid) % len(deviceIDs)])
    else:
        current_device = str(deviceIDs[0])

# Config data loader
data_loader = NodeTrafficLoader(dataset=args['Dataset'], city=args['City'],
                                data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                closeness_len=int(args['CT']), period_len=int(args['PT']), trend_len=int(args['TT']),
                                threshold_interaction=args['TI'], threshold_distance=args['TD'],
                                threshold_correlation=args['TC'], graph=args['Graph'], with_lm=True, normalize=True, MergeIndex=args['MergeIndex'],
                                MergeWay="max" if args["Dataset"] == "ChargeStation" else "sum")

ST_MGCN_Obj = ST_MGCN(T=int(args['CT']) + int(args['PT']) + int(args['TT']),
                      input_dim=1,
                      external_dim=data_loader.external_dim,
                      num_graph=data_loader.LM.shape[0],
                      gcl_k=args['K'],
                      gcl_l=args['L'],
                      lstm_units=args['LSTMUnits'],
                      lstm_layers=args['LSTMLayers'],
                      lr=args['lr'],
                      code_version=code_version,
                      model_dir=model_dir,
                      gpu_device=current_device)

ST_MGCN_Obj.build()

print(args['Dataset'], args['City'], code_version)
print(ST_MGCN_Obj.trainable_vars)

# Training
if args['Train'] == 'True':
    ST_MGCN_Obj.fit(traffic_flow=np.concatenate((np.transpose(data_loader.train_closeness, [0, 2, 1, 3]),
                                                 np.transpose(data_loader.train_period, [0, 2, 1, 3]),
                                                 np.transpose(data_loader.train_trend, [0, 2, 1, 3])), axis=1),
                    laplace_matrix=data_loader.LM,
                    target=data_loader.train_y,
                    external_feature= None,
                    early_stop_method='t-test',
                    output_names=('loss', ),
                    evaluate_loss_name='loss',
                    early_stop_length=int(args['ESlength']),
                    early_stop_patience=float(args['patience']),
                    sequence_length=data_loader.train_sequence_len,
                    save_model=True,
                    batch_size=int(args['BatchSize']))

ST_MGCN_Obj.load(code_version)

# Evaluate
prediction = ST_MGCN_Obj.predict(traffic_flow=np.concatenate((np.transpose(data_loader.test_closeness, [0, 2, 1, 3]),
                                                             np.transpose(data_loader.test_period, [0, 2, 1, 3]),
                                                             np.transpose(data_loader.test_trend, [0, 2, 1, 3])), axis=1),
                                 laplace_matrix=data_loader.LM,
                                 external_feature=None,
                                 sequence_length=data_loader.test_sequence_len,
                                 output_names=['prediction'],
                                 cache_volume=int(args['BatchSize']))

test_rmse = metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                        target=data_loader.normalizer.min_max_denormal(data_loader.test_y),
                        threshold=0)

print('Test result', test_rmse)

val_loss = ST_MGCN_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])

best_val_loss = data_loader.normalizer.min_max_denormal(best_val_loss)

print('Best val result', best_val_loss)

time_consumption = [val_loss[e][0] - val_loss[e-1][0] for e in range(1, len(val_loss))]
time_consumption = sum([e for e in time_consumption if e < (min(time_consumption) * 10)]) / 3600
print('Converged using %.2f hour / %s epochs' % (time_consumption, ST_MGCN_Obj._global_step))

if nni_params:
    nni.report_final_result({
        'default': best_val_loss,
        'test-rmse': test_rmse,
    })