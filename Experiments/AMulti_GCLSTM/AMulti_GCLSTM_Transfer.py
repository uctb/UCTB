import os
import yaml
import argparse
import GPUtil
import numpy as np

from UCTB.dataset import TransferDataLoader, NodeTrafficLoader
from UCTB.model import AMulti_GCLSTM
from UCTB.evaluation import metric

#####################################################################
# argument parser
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('-m', '--model', default='amulti_gclstm_v4.model.yml')
parser.add_argument('-sd', '--source_data', default='bike_nyc.data.yml')
parser.add_argument('-td', '--target_data', default='bike_dc.data.yml')

yml_files = vars(parser.parse_args())

with open(yml_files['model'], 'r') as f:
    model_params = yaml.load(f)

with open(yml_files['source_data'], 'r') as f:
    sd_params = yaml.load(f)

with open(yml_files['target_data'], 'r') as f:
    td_params = yaml.load(f)

assert sd_params['closeness_len'] == td_params['closeness_len']
assert sd_params['period_len'] == td_params['period_len']
assert sd_params['trend_len'] == td_params['trend_len']

#####################################################################
# Generate code_version
group = 'TransferLearning'
code_version = 'AMultiGCLSTM_{}_{}_{}'.format(yml_files['source_data'].split('.')[0],
                                              yml_files['target_data'].split('.')[0], 'V0')

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')
model_dir_path = os.path.join(model_dir_path, group)
#####################################################################
# Config data loader

pre_train = False
fine_tune = True
transfer = True

tl_data_loader = TransferDataLoader(sd_params, td_params, model_params)

deviceIDs = GPUtil.getAvailable(order='first', limit=2, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    current_device = str(deviceIDs[0])

sd_model = AMulti_GCLSTM(num_node=tl_data_loader.sd_loader.station_number,
                         num_graph=tl_data_loader.sd_loader.LM.shape[0],
                         external_dim=tl_data_loader.sd_loader.external_dim,
                         tpe_dim=None if hasattr(tl_data_loader.sd_loader, 'tpe_dim') is False
                         else tl_data_loader.sd_loader.tpe_dim,
                         code_version=code_version,
                         model_dir=model_dir_path,
                         gpu_device=current_device,
                         **sd_params, **model_params)
sd_model.build()

td_model = AMulti_GCLSTM(num_node=tl_data_loader.td_loader.station_number,
                         num_graph=tl_data_loader.td_loader.LM.shape[0],
                         external_dim=tl_data_loader.td_loader.external_dim,
                         tpe_dim=None if hasattr(td_params, 'tpe_dim') is False else td_params.tpe_dim,
                         code_version=code_version,
                         model_dir=model_dir_path,
                         gpu_device=current_device,
                         **td_params, **model_params)
td_model.build()

sd_denormalizer = (lambda x: x) if sd_params['normalize'] is False else tl_data_loader.sd_loader.normalizer.min_max_denormal
td_denormalizer = (lambda x: x) if td_params['normalize'] is False else tl_data_loader.td_loader.normalizer.min_max_denormal

print('#################################################################')
print('Source Domain information')
print(sd_params['dataset'], sd_params['city'], code_version)
print('Number of trainable variables', sd_model.trainable_vars)

print('#################################################################')
print('Target Domain information')
print(td_params['dataset'], td_params['city'], code_version)
print('Number of trainable variables', td_model.trainable_vars)

# Training
if pre_train:
    sd_model.fit(closeness_feature=tl_data_loader.sd_loader.train_closeness,
                 period_feature=tl_data_loader.sd_loader.train_period,
                 trend_feature=tl_data_loader.sd_loader.train_trend,
                 laplace_matrix=tl_data_loader.sd_loader.LM,
                 target=tl_data_loader.sd_loader.train_y,
                 external_feature=tl_data_loader.sd_loader.train_ef,
                 sequence_length=tl_data_loader.sd_loader.train_sequence_len,
                 output_names=('loss',),
                 evaluate_loss_name='loss',
                 op_names=('train_op',),
                 batch_size=sd_params['batch_size'],
                 max_epoch=sd_params['max_epoch'],
                 validate_ratio=0.1,
                 early_stop_method='t-test',
                 early_stop_length=sd_params['early_stop_length'],
                 early_stop_patience=sd_params['early_stop_patience'],
                 verbose=True,
                 save_model=True)

sd_model.load(code_version)
prediction = sd_model.predict(closeness_feature=tl_data_loader.sd_loader.test_closeness,
                              period_feature=tl_data_loader.sd_loader.test_period,
                              trend_feature=tl_data_loader.sd_loader.test_trend,
                              laplace_matrix=tl_data_loader.sd_loader.LM,
                              target=tl_data_loader.sd_loader.test_y,
                              external_feature=tl_data_loader.sd_loader.test_ef,
                              output_names=('prediction',),
                              sequence_length=tl_data_loader.sd_loader.test_sequence_len,
                              cache_volume=sd_params['batch_size'], )

test_prediction = prediction['prediction']

test_rmse, test_mape = metric.rmse(prediction=sd_denormalizer(test_prediction),
                                   target=sd_denormalizer(tl_data_loader.sd_loader.test_y), threshold=0), \
                       metric.mape(prediction=sd_denormalizer(test_prediction),
                                   target=sd_denormalizer(tl_data_loader.sd_loader.test_y), threshold=0)

print('#################################################################')
print('Source Domain Result')
print(test_rmse, test_mape)

td_model.load(code_version)

prediction = td_model.predict(closeness_feature=tl_data_loader.td_loader.test_closeness,
                              period_feature=tl_data_loader.td_loader.test_period,
                              trend_feature=tl_data_loader.td_loader.test_trend,
                              laplace_matrix=tl_data_loader.td_loader.LM,
                              target=tl_data_loader.td_loader.test_y,
                              external_feature=tl_data_loader.td_loader.test_ef,
                              output_names=('prediction',),
                              sequence_length=tl_data_loader.td_loader.test_sequence_len,
                              cache_volume=td_params['batch_size'], )

test_prediction = prediction['prediction']

test_rmse, test_mape = metric.rmse(prediction=td_denormalizer(test_prediction),
                                   target=td_denormalizer(tl_data_loader.td_loader.test_y), threshold=0), \
                       metric.mape(prediction=td_denormalizer(test_prediction),
                                   target=td_denormalizer(tl_data_loader.td_loader.test_y), threshold=0)

print('#################################################################')
print('Target Domain Result')
print(test_rmse, test_mape)

if fine_tune:
    td_model.load(code_version)
    td_model.fit(closeness_feature=tl_data_loader.td_loader.train_closeness,
                 period_feature=tl_data_loader.td_loader.train_period,
                 trend_feature=tl_data_loader.td_loader.train_trend,
                 laplace_matrix=tl_data_loader.td_loader.LM,
                 target=tl_data_loader.td_loader.train_y,
                 external_feature=tl_data_loader.td_loader.train_ef,
                 sequence_length=tl_data_loader.td_loader.train_sequence_len,
                 output_names=('loss',),
                 evaluate_loss_name='loss',
                 op_names=('train_op',),
                 batch_size=td_params['batch_size'],
                 max_epoch=10000,
                 validate_ratio=0.1,
                 early_stop_method='t-test',
                 early_stop_length=td_params['early_stop_length'],
                 early_stop_patience=td_params['early_stop_patience'],
                 verbose=True,
                 save_model=False)

    prediction = td_model.predict(closeness_feature=tl_data_loader.td_loader.test_closeness,
                                  period_feature=tl_data_loader.td_loader.test_period,
                                  trend_feature=tl_data_loader.td_loader.test_trend,
                                  laplace_matrix=tl_data_loader.td_loader.LM,
                                  target=tl_data_loader.td_loader.test_y,
                                  external_feature=tl_data_loader.td_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=tl_data_loader.td_loader.test_sequence_len,
                                  cache_volume=td_params['batch_size'], )

    test_prediction = prediction['prediction']

    test_rmse, test_mape = metric.rmse(prediction=td_denormalizer(test_prediction),
                                       target=td_denormalizer(tl_data_loader.td_loader.test_y), threshold=0), \
                           metric.mape(prediction=td_denormalizer(test_prediction),
                                       target=td_denormalizer(tl_data_loader.td_loader.test_y), threshold=0)

    print('#################################################################')
    print('Target Domain Fine-tune')
    print(test_rmse, test_mape)

if transfer:
    td_model.load(code_version)
    traffic_sim = tl_data_loader.traffic_sim()
    transfer_epoch = 20000

    # prepare data:
    feature_maps = []
    for record in traffic_sim:
        # score, index, start, end
        sd_transfer_data = tl_data_loader.sd_loader.train_data[record[2]: record[3], :]

        st_transfer_closeness, \
        st_transfer_period, \
        st_transfer_trend, \
        _ = tl_data_loader.sd_loader.st_move_sample.move_sample(sd_transfer_data)

        fm = sd_model.predict(closeness_feature=st_transfer_closeness,
                              period_feature=st_transfer_period,
                              trend_feature=st_transfer_trend,
                              laplace_matrix=tl_data_loader.sd_loader.LM,
                              external_feature=tl_data_loader.sd_loader.train_ef,
                              output_names=['feature_map'],
                              sequence_length=len(st_transfer_closeness),
                              cache_volume=sd_params['batch_size'])

        feature_maps.append(fm['feature_map'][:, record[1]:record[1] + 1, :, :])

    feature_maps = np.concatenate(feature_maps, axis=1)

    # transfer
    td_model.fit(closeness_feature=tl_data_loader.td_loader.train_closeness,
                 period_feature=tl_data_loader.td_loader.train_period,
                 trend_feature=tl_data_loader.td_loader.train_trend,
                 laplace_matrix=tl_data_loader.td_loader.LM,
                 target=tl_data_loader.td_loader.train_y,
                 external_feature=tl_data_loader.td_loader.train_ef,
                 similar_feature_map=feature_maps,
                 sequence_length=tl_data_loader.td_loader.train_sequence_len,
                 output_names=('transfer_loss',),
                 evaluate_loss_name='transfer_loss',
                 op_names=('transfer_op',),
                 batch_size=td_params['batch_size'],
                 max_epoch=transfer_epoch,
                 validate_ratio=0.1,
                 early_stop_method='t-test',
                 early_stop_length=td_params['early_stop_length'],
                 early_stop_patience=td_params['early_stop_patience'],
                 verbose=True,
                 save_model=False)

    prediction = td_model.predict(closeness_feature=tl_data_loader.td_loader.test_closeness,
                                  period_feature=tl_data_loader.td_loader.test_period,
                                  trend_feature=tl_data_loader.td_loader.test_trend,
                                  laplace_matrix=tl_data_loader.td_loader.LM,
                                  target=tl_data_loader.td_loader.test_y,
                                  external_feature=tl_data_loader.td_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=tl_data_loader.td_loader.test_sequence_len,
                                  cache_volume=td_params['batch_size'], )

    test_prediction = prediction['prediction']

    test_rmse, test_mape = metric.rmse(prediction=td_denormalizer(test_prediction),
                                       target=td_denormalizer(tl_data_loader.td_loader.test_y), threshold=0), \
                           metric.mape(prediction=td_denormalizer(test_prediction),
                                       target=td_denormalizer(tl_data_loader.td_loader.test_y), threshold=0)

    print('#################################################################')
    print('Target Domain Transfer')
    print(test_rmse, test_mape)