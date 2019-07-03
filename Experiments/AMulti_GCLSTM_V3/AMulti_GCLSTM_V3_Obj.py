import os
import nni
import copy
import yaml
import argparse
import GPUtil
import hashlib

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import AMulti_GCLSTM_V3
from UCTB.evaluation import metric
from UCTB.preprocess.time_utils import is_work_day_chine, is_work_day_america

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')

parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('-y', default='metro_shanghai.yml')

yml_file = parser.parse_args()

with open(yml_file.y, 'r') as f:
    args = yaml.load(f)

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)

#####################################################################
# Generate code_version
params_hash = copy.deepcopy(args)
for key in ['train', 'max_epoch', 'gpu_device', 'code_version']:
    params_hash.pop(key)
params_hash = hashlib.md5(str(sorted(params_hash.items(), key=lambda x: x[0], reverse=False)).encode()).hexdigest()

model_dir = os.path.join(model_dir_path, args['group'])
# code_version = 'AMultiGCLSTM_V3_{}_K{}L{}_{}'.format(''.join([e[0] for e in args['graph'].split('-')]),
#                                                      args['gcn_k'], args['gcn_layers'], code_version)

code_version = 'AM{}'.format(params_hash)

# Config data loader
data_loader = NodeTrafficLoader(dataset=args['dataset'], city=args['city'],
                                data_range=args['data_range'], train_data_length=args['train_day_length'],
                                test_ratio=0.1,
                                closeness_len=args['closeness_len'],
                                period_len=args['period_len'],
                                trend_len=args['trend_len'],
                                threshold_distance=args['threshold_distance'],
                                threshold_correlation=args['threshold_correlation'],
                                threshold_interaction=args['threshold_interaction'],
                                normalize=True if args['normalize'] == 'True' else False,
                                graph=args['graph'],
                                with_lm=True, with_tpe=True if args['st_method'] == 'gal_gcn' else False,
                                workday_parser=is_work_day_america if args['dataset'] == 'Bike' else is_work_day_chine)

de_normalizer = None if args['normalize'] == 'False' else data_loader.normalizer.min_max_denormal

deviceIDs = GPUtil.getAvailable(order = 'first', limit = 2, maxLoad = 0.3, maxMemory = 0.3,
                                includeNan=False, excludeID=[], excludeUUID=[])

CPT_AMulti_GCLSTM_Obj = AMulti_GCLSTM_V3(num_node=data_loader.station_number,
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
                                         tpe_dim=1,
                                         temporal_gal_units=args['temporal_gal_units'],
                                         temporal_gal_num_heads=args['temporal_gal_num_heads'],
                                         temporal_gal_layers=args['temporal_gal_layers'],
                                         # merge parameters
                                         graph_merge_gal_units=args['graph_merge_gal_units'],
                                         graph_merge_gal_num_heads=args['graph_merge_gal_num_heads'],
                                         temporal_merge_gal_units=args['temporal_merge_gal_units'],
                                         temporal_merge_gal_num_heads=args['temporal_merge_gal_num_heads'],
                                         # network structure parameters
                                         st_method=args['st_method'],          # gclstm
                                         temporal_merge=args['temporal_merge'],     # gal
                                         graph_merge=args['graph_merge'],        # concat
                                         lr=float(args['lr']),
                                         code_version=code_version,
                                         model_dir=model_dir,
                                         gpu_device='0')
                                         # gpu_device=str(deviceIDs[int(nni_sid) % len(deviceIDs)]))

CPT_AMulti_GCLSTM_Obj.build()

print(args['dataset'], args['city'], code_version)
print('Number of trainable variables', CPT_AMulti_GCLSTM_Obj.trainable_vars)

# # Training
if args['train'] == 'True':
    CPT_AMulti_GCLSTM_Obj.fit(closeness_feature=data_loader.train_closeness,
                              period_feature=data_loader.train_period,
                              trend_feature=data_loader.train_trend,
                              laplace_matrix=data_loader.LM,
                              target=data_loader.train_y,
                              external_feature=data_loader.train_ef,
                              early_stop_method='t-test',
                              early_stop_length=args['early_stop_length'],
                              early_stop_patience=args['early_stop_patience'],
                              max_epoch=args['max_epoch'],
                              batch_size=args['batch_size'])

CPT_AMulti_GCLSTM_Obj.load(code_version)

# Evaluate
test_error = CPT_AMulti_GCLSTM_Obj.evaluate(closeness_feature=data_loader.test_closeness,
                                            period_feature=data_loader.test_period,
                                            trend_feature=data_loader.test_trend,
                                            laplace_matrix=data_loader.LM,
                                            target=data_loader.test_y,
                                            external_feature=data_loader.test_ef,
                                            cache_volume=args['batch_size'],
                                            metrics=[metric.rmse, metric.mape],
                                            de_normalizer=de_normalizer,
                                            threshold=0)

test_prediction = CPT_AMulti_GCLSTM_Obj.predict(closeness_feature=data_loader.test_closeness,
                                                period_feature=data_loader.test_period,
                                                trend_feature=data_loader.test_trend,
                                                external_feature=data_loader.test_ef,
                                                laplace_matrix=data_loader.LM)

val_loss = CPT_AMulti_GCLSTM_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])

print('Best val result', best_val_loss)
print('Test result', test_error)

if nni_params:
    nni.report_final_result({
        'default': best_val_loss,
        'test-rmse': test_error[0],
        'test-mape': test_error[1]
    })

# def show_prediction(prediction, target, station_index, start=0, end=-1):
#
#     import matplotlib.pyplot as plt
#
#     # fig, axs = plt.subplots(1, 2, figsize=(9, 3))
#     # axs[0].plot(prediction[start:end, station_index])
#     # axs[1].plot(target[start:end, station_index])
#
#     plt.plot(prediction[start:end, station_index], 'b')
#     plt.plot(target[start:end, station_index], 'r')
#
#     print(metric.rmse(prediction[start:end, station_index], target[start:end, station_index]))
#
#     print(prediction[start:end, station_index].max(), target[start:end, station_index].max())
#     print(prediction[start:end, station_index].min(), target[start:end, station_index].min())
#
#     plt.show()
#
# show_prediction(test_prediction, data_loader.test_y, station_index=10)