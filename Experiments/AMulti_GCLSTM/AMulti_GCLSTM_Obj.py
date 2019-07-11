import os
import nni
import yaml
import argparse
import GPUtil

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import AMulti_GCLSTM
from UCTB.evaluation import metric
from UCTB.preprocess.time_utils import is_work_day_china, is_work_day_america

#####################################################################
# argument parser
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('-m', '--model', default='amulti_gclstm_v4.model.yml')
parser.add_argument('-d', '--data', default='didi_chengdu.data.yml')

yml_files = vars(parser.parse_args())

args = {}
for _, yml_file in yml_files.items():
    with open(yml_file, 'r') as f:
        args.update(yaml.load(f))

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)

#####################################################################
# Generate code_version
if nni_params:
    args['mark'] += str(nni_sid)
code_version = 'AMultiGCLSTM_{}_{}_K{}L{}_{}'.format(args['model_version'],
                                                     ''.join([e[0] for e in args['graph'].split('-')]),
                                                     args['gcn_k'], args['gcn_layers'], args['mark'])
model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')
model_dir_path = os.path.join(model_dir_path, args['group'])
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

de_normalizer = None if args['normalize'] is False else data_loader.normalizer.min_max_denormal

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    if nni_params:
        current_device = str(deviceIDs[int(nni_sid) % len(deviceIDs)])
    else:
        current_device = str(deviceIDs[0])

amulti_gclstm_obj = AMulti_GCLSTM(num_node=data_loader.station_number,
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim,
                                  closeness_len=args['closeness_len'],
                                  period_len=args['period_len'],
                                  trend_len=args['trend_len'],
                                  gcn_k=args['gcn_k'],
                                  gcn_layers=args['gcn_layers'],
                                  gclstm_layers=args['gclstm_layers'],
                                  num_hidden_units=args['num_hidden_units'],
                                  num_filter_conv1x1=args['num_filter_conv1x1'],
                                  # temporal attention parameters
                                  tpe_dim=None if hasattr(args, 'tpe_dim') is False else args.tpe_dim,
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
                                  gpu_device=current_device)

amulti_gclstm_obj.build()

print(args['dataset'], args['city'], code_version)
print('Number of trainable variables', amulti_gclstm_obj.trainable_vars)

# # Training
if args['train']:
    amulti_gclstm_obj.fit(closeness_feature=data_loader.train_closeness,
                          period_feature=data_loader.train_period,
                          trend_feature=data_loader.train_trend,
                          laplace_matrix=data_loader.LM,
                          target=data_loader.train_y,
                          external_feature=data_loader.train_ef,
                          sequence_length=data_loader.train_sequence_len,
                          output_names=('loss', ),
                          evaluate_loss_name='loss',
                          op_names=('train_op', ),
                          batch_size=args['batch_size'],
                          max_epoch=args['max_epoch'],
                          validate_ratio=0.1,
                          early_stop_method='t-test',
                          early_stop_length=args['early_stop_length'],
                          early_stop_patience=args['early_stop_patience'],
                          verbose=True,
                          save_model=True)

amulti_gclstm_obj.load(code_version)

prediction = amulti_gclstm_obj.predict(closeness_feature=data_loader.test_closeness,
                                       period_feature=data_loader.test_period,
                                       trend_feature=data_loader.test_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       output_names=('prediction', ),
                                       sequence_length=data_loader.test_sequence_len,
                                       cache_volume=args['batch_size'], )

test_prediction = prediction['prediction']

# if de_normalizer:
#     test_prediction = de_normalizer(test_prediction)
#     data_loader.test_y = de_normalizer(data_loader.test_y)

test_rmse, test_mape = metric.rmse(prediction=test_prediction, target=data_loader.test_y, threshold=0),\
                       metric.mape(prediction=test_prediction, target=data_loader.test_y, threshold=0)

# Evaluate
val_loss = amulti_gclstm_obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])

print('Best val result', best_val_loss)
print('Test result', test_rmse, test_mape)

if nni_params:
    nni.report_final_result({
        'default': best_val_loss,
        'test-rmse': test_rmse,
        'test-mape': test_mape
    })