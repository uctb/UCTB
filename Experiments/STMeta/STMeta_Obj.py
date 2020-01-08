import os
import nni
import yaml
import argparse
import GPUtil

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import STMeta
from UCTB.model_unit import GraphBuilder
from UCTB.evaluation import metric
from UCTB.preprocess.time_utils import is_work_day_china, is_work_day_america
from UCTB.preprocess import SplitData
import sys
sys.path.append("../")

#from RoadDataLoader import RoadDataLoader
from sendInfo import *
#####################################################################
# argument parser
parser = argparse.ArgumentParser(description="Argument Parser")


parser.add_argument('-l', '--decay_param', default = None)

parser.add_argument('-m', '--model', default = 'STMeta_v1.model.yml')
parser.add_argument('-d', '--data', default = 'DiDi_RoadTTI.yml')
parser.add_argument('-p', '--update_params', default = 'gcn_k:1,gclstm_layers:1,batch_size:16,mark:TEST')

# Parse params
terminal_vars=vars(parser.parse_args())

# decay param
if terminal_vars['decay_param'] is not None:
    terminal_vars['decay_param'] = os.path.join(os.getcwd(),terminal_vars['decay_param'])

yml_files=[terminal_vars['model'], terminal_vars['data']]
args={}
for yml_file in yml_files:
    with open(yml_file, 'r') as f:
        args.update(yaml.load(f))

if len(terminal_vars['update_params']) > 0:
    args.update({e.split(':')[0]: e.split(':')[1]
                for e in terminal_vars['update_params'].split(',')})
    print({e.split(':')[0]: e.split(':')[1]
          for e in terminal_vars['update_params'].split(',')})

nni_params=nni.get_next_parameter()
nni_sid=nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    args['mark'] += str(nni_sid)


# to make sure its type is int
args['closeness_len'] = int(args['closeness_len'])
args['period_len'] = int(args['period_len'])
args['trend_len'] = int(args['trend_len'])
#####################################################################
# Generate code_version
code_version='{}_C{}P{}T{}_G{}_K{}L{}_{}'.format(args['model_version'],
                                                   args['closeness_len'], args['period_len'],
                                                   args['trend_len'],
                                                   ''.join(
                                                       [e[0] for e in args['graph'].split('-')]),
                                                   args['gcn_k'], args['gcn_layers'], args['mark'])
model_dir_path=os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'model_dir')
model_dir_path=os.path.join(model_dir_path, args['group'])
#####################################################################

# Config data loader
data_loader = NodeTrafficLoader(dataset=args['dataset'], city=args['city'],
                                data_range=args['data_range'], train_data_length=args['train_data_length'],
                                test_ratio=0.1,
                                closeness_len=args['closeness_len'],
                                period_len=args['period_len'],
                                trend_len=args['trend_len'],
                                threshold_distance=args['threshold_distance'],
                                threshold_correlation=args['threshold_correlation'],
                                threshold_interaction=args['threshold_interaction'],
                                normalize=args['normalize'],
                                graph=args['graph'],
                                with_lm=True, with_tpe=True if args['st_method'] == 'gal_gcn' else False,
                                workday_parser=is_work_day_america if args['dataset'] == 'Bike' else is_work_day_china)



# split data
train_closeness, val_closeness = SplitData.split_data(data_loader.train_closeness, [0.9, 0.1])
train_period, val_period = SplitData.split_data(data_loader.train_period, [0.9, 0.1])
train_trend, val_trend = SplitData.split_data(data_loader.train_trend, [0.9, 0.1])
train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])
train_ef, val_ef = SplitData.split_data(data_loader.train_ef, [0.9, 0.1])


de_normalizer = None if args['normalize'] is False else data_loader.normalizer.min_max_denormal

deviceIDs = GPUtil.getAvailable(order='load', limit=2, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    if nni_params:
        current_device = str(deviceIDs[int(nni_sid) % len(deviceIDs)])
    else:
        current_device = str(deviceIDs[0])

STMeta_obj = STMeta(num_node=data_loader.station_number,
                                 num_graph=data_loader.LM.shape[0],
                                 external_dim=data_loader.external_dim,
                                 closeness_len=args['closeness_len'],
                                 period_len=args['period_len'],
                                 trend_len=args['trend_len'],
                                 gcn_k=int(args['gcn_k']),
                                 gcn_layers=int(args['gcn_layers']),
                                 gclstm_layers=int(args['gclstm_layers']),
                                 num_hidden_units=args['num_hidden_units'],
                                 num_filter_conv1x1=args['num_filter_conv1x1'],
                                 # temporal attention parameters
                                 tpe_dim=data_loader.tpe_dim,
                                 temporal_gal_units=args.get('temporal_gal_units'),
                                 temporal_gal_num_heads=args.get('temporal_gal_num_heads'),
                                 temporal_gal_layers=args.get('temporal_gal_layers'),
                                 # merge parameters
                                 graph_merge_gal_units=args['graph_merge_gal_units'],
                                 graph_merge_gal_num_heads=args['graph_merge_gal_num_heads'],
                                 temporal_merge_gal_units=args['temporal_merge_gal_units'],
                                 temporal_merge_gal_num_heads=args['temporal_merge_gal_num_heads'],
                                 # network structure parameters
                                 st_method=args['st_method'],  # gclstm
                                 temporal_merge=args['temporal_merge'],  # gal
                                 graph_merge=args['graph_merge'],  # concat
                                 build_transfer=args['build_transfer'],
                                 lr=float(args['lr']),
                                 code_version=code_version,
                                 model_dir=model_dir_path,
                                 gpu_device=current_device,
                                 embedding_flag=args['embedding_flag'],
                                 embedding_dim =args['embedding_dim'],
                                 classified_embedding=data_loader.external_onehot_dim if args['classified_embedding'] is True else [],
                                 decay_param=terminal_vars['decay_param'])

STMeta_obj.build()

print(args['dataset'], code_version)
print('Number of trainable variables', STMeta_obj.trainable_vars)
print('Number of training samples', data_loader.train_sequence_len)

# # Training
if args['train']:
    STMeta_obj.fit(closeness_feature=data_loader.train_closeness,
                          period_feature=data_loader.train_period,
                          trend_feature=data_loader.train_trend,
                          laplace_matrix=data_loader.LM,
                          target=data_loader.train_y,
                          external_feature=data_loader.train_ef,
                          sequence_length=data_loader.train_sequence_len,
                          output_names=('loss','lr'),
                          evaluate_loss_name='loss',
                          op_names=('train_op', ),
                          batch_size=int(args['batch_size']),
                          max_epoch=int(args['max_epoch']),
                          validate_ratio=0.1,
                          early_stop_method='naive',
                          early_stop_length=args['early_stop_length'],
                          early_stop_patience=args['early_stop_patience'],
                          verbose=True,
                          save_model=True)

STMeta_obj.load(code_version)




prediction = STMeta_obj.predict(closeness_feature=data_loader.test_closeness,
                                       period_feature=data_loader.test_period,
                                       trend_feature=data_loader.test_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       output_names=('prediction', ),
                                       sequence_length=data_loader.test_sequence_len,
                                       cache_volume=int(args['batch_size']), )

test_prediction = prediction['prediction']

if de_normalizer:
    test_prediction = de_normalizer(test_prediction)
    data_loader.test_y = de_normalizer(data_loader.test_y)

test_rmse = metric.rmse(prediction=test_prediction, target=data_loader.test_y, threshold=0)

# val pred
prediction = STMeta_obj.predict(closeness_feature=val_closeness,
                                       period_feature=val_period,
                                       trend_feature=val_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=val_y,
                                       external_feature=val_ef,
                                       output_names=('prediction', ),
                                       sequence_length=max((len(val_closeness), len(val_period), len(val_trend))),
                                       cache_volume=int(args['batch_size']), )

val_prediction = prediction['prediction']

if de_normalizer:
    val_prediction = de_normalizer(val_prediction)
    val_y = de_normalizer(val_y)

val_rmse = metric.rmse(prediction=val_prediction, target=val_y, threshold=0)
 


# # Evaluate
val_loss = STMeta_obj.load_event_scalar('val_loss')
# best_val_loss = min([e[-1] for e in val_loss])

# if de_normalizer:
#     best_val_loss = de_normalizer(best_val_loss)



print('Val result', val_rmse )
print('Test result', test_rmse)

time_consumption = [val_loss[e][0] - val_loss[e-1][0] for e in range(1, len(val_loss))]
time_consumption = sum([e for e in time_consumption if e < (min(time_consumption) * 10)]) / 3600
print('Converged using %.2f hour / %s epochs' % (time_consumption, STMeta_obj._global_step))

#senInfo("{},已完成，请尽快查看~".format(code_version))

if nni_params:
    nni.report_final_result({
        'default': val_rmse,
        'test-rmse': test_rmse,
    })
