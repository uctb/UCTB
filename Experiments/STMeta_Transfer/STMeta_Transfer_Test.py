import os
import yaml
import argparse
import GPUtil
import numpy as np

from UCTB.dataset import TransferDataLoader
from UCTB.model import STMeta
from UCTB.evaluation import metric
from UCTB.train import EarlyStoppingTTest
from UCTB.preprocess.GraphGenerator import GraphGenerator

#####################################################################
# argument parser
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('-m', '--model', default='STMeta_v4.model.yml')
parser.add_argument('-sd', '--source_data', default='bike_nyc.data.yml')
parser.add_argument('-td', '--target_data', default='bike_chicago.data.yml')
parser.add_argument('-smd', '--similarity_mode', default='checkin')  # 'checkin', 'traffic', 'fake_traffic', 'poi'
parser.add_argument('-tdl', '--target_data_length', default='1', type=str)
parser.add_argument('-tfr', '--transfer_ratio', default='0.1', type=str)
parser.add_argument('-pt', '--pretrain', default='True')
parser.add_argument('-ft', '--finetune', default='True')
parser.add_argument('-tr', '--transfer', default='True')

dynamic_mode = False

sd_regularization = True

validate_mode = 'val'  # test

val_ratio = 0.3

args = vars(parser.parse_args())

similarity_mode = args['similarity_mode']  # 'checkin', 'traffic', 'fake_traffic', 'poi'

with open(args['model'].strip('./\\'), 'r') as f:
    model_params = yaml.load(f)

with open(args['source_data'].strip('./\\'), 'r') as f:
    sd_params = yaml.load(f)

with open(args['target_data'].strip('./\\'), 'r') as f:
    td_params = yaml.load(f)

assert sd_params['closeness_len'] == td_params['closeness_len']
assert sd_params['period_len'] == td_params['period_len']
assert sd_params['trend_len'] == td_params['trend_len']


def show_prediction(pretrain, finetune, transfer, target, station_index, start=0, end=-1):

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()

    axs.plot(pretrain[start:end, station_index], 'b', label='pretrain')
    axs.plot(finetune[start:end, station_index], 'g', label='finetune')
    axs.plot(transfer[start:end, station_index], 'y', label='transfer')
    axs.plot(target[start:end, station_index], 'r', label='target')

    axs.grid()
    axs.legend(fontsize=15)

    axs.set_xlabel('Time', fontsize=15)
    axs.set_ylabel('Demand', fontsize=15)

    axs.set_title('Station/Grid %s' % station_index)

    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PNG'),
                             '%s.png' % 'station-%s' % station_index), dpi=150)
    plt.close()

#####################################################################
# Generate code_version
group = 'STMeta_Transfer'
code_version = 'STMeta_SD_{}_TD_{}'.format(args['source_data'].split('.')[0].split('_')[-1],
                                                 args['target_data'].split('.')[0].split('_')[-1])

sub_code_version = 'SR_C{}P{}T{}_G{}'.format(sd_params['closeness_len'], sd_params['period_len'], sd_params['trend_len'],
                                             ''.join([e[0].upper() for e in sd_params['graph'].split('-')]))

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')
model_dir_path = os.path.join(model_dir_path, group)
#####################################################################
# Config data loader

data_loader = TransferDataLoader(sd_params, td_params, model_params, td_data_length=args['target_data_length'])

# Import the Class:GraphGenerator
# Call GraphGenerator to initialize and generate LM
graph = sd_params['graph']
sd_graphBuilder = GraphGenerator(graph,
                             dataset = data_loader.sd_loader.dataset,
                             train_data = data_loader.sd_loader.train_data,
                             traffic_data_index = data_loader.sd_loader.traffic_data_index,
                             train_test_ratio = data_loader.sd_loader.train_test_ratio,
                             threshold_distance=sd_params['threshold_distance'],
                             threshold_correlation=sd_params['threshold_correlation'],
                             threshold_interaction=sd_params['threshold_interaction'],
                             )

graph = td_params['graph']
td_graphBuilder = GraphGenerator(graph,
                             dataset = data_loader.td_loader.dataset,
                             train_data = data_loader.td_loader.train_data,
                             traffic_data_index = data_loader.td_loader.traffic_data_index,
                             train_test_ratio = data_loader.td_loader.train_test_ratio,
                             threshold_distance=td_params['threshold_distance'],
                             threshold_correlation=td_params['threshold_correlation'],
                             threshold_interaction=td_params['threshold_interaction'],
                             )

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=1,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    current_device = str(deviceIDs[0])

sd_model = STMeta(num_node=data_loader.sd_loader.station_number,
                  num_graph=sd_graphBuilder.LM.shape[0],
                  external_dim=data_loader.sd_loader.external_dim,
                  tpe_dim=data_loader.sd_loader.tpe_dim,
                  code_version=code_version,
                  model_dir=model_dir_path,
                  gpu_device=current_device,
                  build_sd_regularization=sd_regularization,
                  transfer_ratio=0,
                  **sd_params, **model_params)
sd_model.build(init_vars=True)

transfer_ratio = float(args['transfer_ratio'])

td_model = STMeta(num_node=data_loader.td_loader.station_number,
                  num_graph=td_graphBuilder.LM.shape[0],
                  external_dim=data_loader.td_loader.external_dim,
                  tpe_dim=data_loader.td_loader.tpe_dim,
                  code_version=code_version,
                  model_dir=model_dir_path,
                  transfer_ratio=transfer_ratio,
                  build_sd_regularization=False,
                  gpu_device=current_device,
                  **td_params, **model_params)

td_model.build(init_vars=False, max_to_keep=None)

sd_de_normalizer = (lambda x: x) if sd_params['normalize'] is False \
                                else data_loader.sd_loader.normalizer.min_max_denormal
td_de_normalizer = (lambda x: x) if td_params['normalize'] is False \
                                else data_loader.td_loader.normalizer.min_max_denormal

print('#################################################################')
print('Source Domain information')
print(sd_params['dataset'], sd_params['city'])
print('Number of trainable variables', sd_model.trainable_vars)
print('Number of training samples', data_loader.sd_loader.train_sequence_len)

print('#################################################################')
print('Target Domain information')
print(td_params['dataset'], td_params['city'])
print('Number of trainable variables', td_model.trainable_vars)
print('Number of training samples', data_loader.td_loader.train_sequence_len)

pretrain_model_name = 'Pretrain_' + sub_code_version
finetune_model_name = 'Finetune_' + sub_code_version + '_' + str(data_loader.td_loader.train_sequence_len)
transfer_model_name = 'Transfer_' + sub_code_version + '_' + str(data_loader.td_loader.train_sequence_len) +\
                      '_%s' % int((transfer_ratio * 100)) + '%' + '_' + str(similarity_mode)

writing_obj = [''.join([e for e in sd_params['graph'].split('-')]),
               similarity_mode,
               args['source_data'].split('.')[0].split('_')[-1],
               args['target_data'].split('.')[0].split('_')[-1],
               str(transfer_ratio), '%s天' % args['target_data_length']]

if args['pretrain'] == 'True':
    try:
        sd_model.load(pretrain_model_name)
    except FileNotFoundError:
        sd_model.fit(closeness_feature=data_loader.sd_loader.train_closeness,
                     period_feature=data_loader.sd_loader.train_period,
                     trend_feature=data_loader.sd_loader.train_trend,
                     laplace_matrix=sd_graphBuilder.LM,
                     target=data_loader.sd_loader.train_y,
                    #  sd_sim=data_loader.checkin_sim_sd(),
                     external_feature=data_loader.sd_loader.train_ef,
                     sequence_length=data_loader.sd_loader.train_sequence_len,
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
        sd_model.save(pretrain_model_name, global_step=0)

    sd_model.load(pretrain_model_name)

    prediction = sd_model.predict(closeness_feature=data_loader.sd_loader.test_closeness,
                                  period_feature=data_loader.sd_loader.test_period,
                                  trend_feature=data_loader.sd_loader.test_trend,
                                  laplace_matrix=sd_graphBuilder.LM,
                                  target=data_loader.sd_loader.test_y,
                                  external_feature=data_loader.sd_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=data_loader.sd_loader.test_sequence_len,
                                  cache_volume=sd_params['batch_size'], )

    test_prediction = prediction['prediction']

    test_rmse, test_mape = metric.rmse(prediction=sd_de_normalizer(test_prediction),
                                       target=sd_de_normalizer(data_loader.sd_loader.test_y)), \
                           metric.mape(prediction=sd_de_normalizer(test_prediction),
                                       target=sd_de_normalizer(data_loader.sd_loader.test_y), threshold=0)

    print('#################################################################')
    print('Source Domain Result')
    print(test_rmse, test_mape)

    td_model.load(pretrain_model_name)

    prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                  period_feature=data_loader.td_loader.test_period,
                                  trend_feature=data_loader.td_loader.test_trend,
                                  laplace_matrix=td_graphBuilder.LM,
                                  target=data_loader.td_loader.test_y,
                                  external_feature=data_loader.td_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=data_loader.td_loader.test_sequence_len,
                                  cache_volume=td_params['batch_size'], )

    pretrain_prediction = prediction['prediction']

    test_rmse, test_mape = metric.rmse(prediction=td_de_normalizer(pretrain_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y)), \
                           metric.mape(prediction=td_de_normalizer(pretrain_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y), threshold=0)

    print('#################################################################')
    print('Target Domain Result')
    print(test_rmse, test_mape)
    writing_obj.append('%.5f' % test_rmse)

if args['finetune'] == 'True':
    try:
        td_model.load(finetune_model_name)
    except FileNotFoundError:
        td_model.load(pretrain_model_name)

        early_stop = EarlyStoppingTTest(td_params['early_stop_length'], td_params['early_stop_patience'])
        best_value = None
        for epoch in range(td_params['max_epoch']):

            output = td_model.fit(closeness_feature=data_loader.td_loader.train_closeness,
                                  period_feature=data_loader.td_loader.train_period,
                                  trend_feature=data_loader.td_loader.train_trend,
                                  laplace_matrix=td_graphBuilder.LM,
                                #   sd_sim=np.array(range(data_loader.td_loader.station_number), dtype=np.int32),
                                  target=data_loader.td_loader.train_y,
                                  external_feature=data_loader.td_loader.train_ef,
                                  sequence_length=data_loader.td_loader.train_sequence_len,
                                  output_names=('loss',),
                                  evaluate_loss_name='loss',
                                  op_names=('train_op',),
                                  batch_size=td_params['batch_size'],
                                  max_epoch=1,
                                  validate_ratio=val_ratio,
                                  early_stop_method='t-test',
                                  early_stop_length=td_params['early_stop_length'],
                                  early_stop_patience=td_params['early_stop_patience'],
                                  verbose=True,
                                  save_model=False,
                                  save_model_name=None,
                                  auto_load_model=False,
                                  return_outputs=True)

            prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                          period_feature=data_loader.td_loader.test_period,
                                          trend_feature=data_loader.td_loader.test_trend,
                                          laplace_matrix=td_graphBuilder.LM,
                                          target=data_loader.td_loader.test_y,
                                          external_feature=data_loader.td_loader.test_ef,
                                          output_names=('prediction',),
                                          sequence_length=data_loader.td_loader.test_sequence_len,
                                          cache_volume=td_params['batch_size'], )

            test_rmse = metric.rmse(prediction=td_de_normalizer(prediction['prediction']),
                                    target=td_de_normalizer(data_loader.td_loader.test_y))

            validate_error = output[-1]['val_loss'] if validate_mode == 'val' else test_rmse

            if early_stop.stop(validate_error):
                break

            if best_value is None or best_value > validate_error:
                best_value = validate_error
                td_model.save(finetune_model_name, global_step=0)

            print(epoch, 'test rmse', test_rmse)

    td_model.load(finetune_model_name)
    prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                  period_feature=data_loader.td_loader.test_period,
                                  trend_feature=data_loader.td_loader.test_trend,
                                  laplace_matrix=td_graphBuilder.LM,
                                  target=data_loader.td_loader.test_y,
                                  external_feature=data_loader.td_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=data_loader.td_loader.test_sequence_len,
                                  cache_volume=td_params['batch_size'], )
    finetune_prediction = prediction['prediction']
    finetune_error_station = np.array([metric.rmse(td_de_normalizer(finetune_prediction[:, i]),
                                                   td_de_normalizer(data_loader.td_loader.test_y[:, i]))
                                                   for i in range(len(finetune_prediction[0]))])
    test_rmse, test_mape = metric.rmse(prediction=td_de_normalizer(finetune_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y)), \
                           metric.mape(prediction=td_de_normalizer(finetune_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y), threshold=0)

    print('#################################################################')
    print('Target Domain Fine-tune')
    print(test_rmse, test_mape)
    writing_obj.append('%.5f' % test_rmse)

if args['transfer'] == 'True':

    try:
        td_model.load(transfer_model_name)
    except FileNotFoundError:

        sd_model.load(pretrain_model_name)
        sd_model.save(transfer_model_name, global_step=0)
        sd_transfer_data = data_loader.sd_loader.train_data[-data_loader.td_loader.train_data.shape[0]:, :]
        transfer_closeness, \
        transfer_period, \
        transfer_trend, \
        _ = data_loader.sd_loader.st_move_sample.move_sample(sd_transfer_data)

        if similarity_mode == 'checkin':
            traffic_sim = data_loader.checkin_sim()
        elif similarity_mode == 'poi':
            traffic_sim = data_loader.poi_sim()
        elif similarity_mode == 'traffic':
            traffic_sim = data_loader.traffic_sim()
        elif similarity_mode == 'fake_traffic':
            traffic_sim = data_loader.traffic_sim_fake()

        def dynamic_fm():
            sd_model.load(transfer_model_name)
            fm = sd_model.predict(closeness_feature=transfer_closeness,
                                  period_feature=transfer_period,
                                  trend_feature=transfer_trend,
                                  laplace_matrix=sd_graphBuilder.LM,
                                  external_feature=data_loader.sd_loader.train_ef,
                                  output_names=['feature_map'],
                                  sequence_length=len(transfer_closeness),
                                  cache_volume=sd_params['batch_size'])
            return np.take(fm['feature_map'], np.array([e[1] for e in traffic_sim]), axis=1)

        feature_maps = dynamic_fm()

        early_stop = EarlyStoppingTTest(td_params['early_stop_length'], td_params['early_stop_patience'])
        td_model.load(pretrain_model_name)
        best_value = None
        for epoch in range(td_params['max_epoch']):

            output = td_model.fit(closeness_feature=data_loader.td_loader.train_closeness,
                                  period_feature=data_loader.td_loader.train_period,
                                  trend_feature=data_loader.td_loader.train_trend,
                                #   sd_sim=np.array(range(data_loader.td_loader.station_number), dtype=np.int32),
                                  laplace_matrix=td_graphBuilder.LM,
                                  target=data_loader.td_loader.train_y,
                                  external_feature=data_loader.td_loader.train_ef,
                                  similar_feature_map=feature_maps,
                                  sequence_length=data_loader.td_loader.train_sequence_len,
                                  output_names=('transfer_loss', 'loss'),
                                  evaluate_loss_name='loss',
                                  op_names=('transfer_op',),
                                  batch_size=td_params['batch_size'],
                                  max_epoch=1,
                                  validate_ratio=val_ratio,
                                  early_stop_method='t-test',
                                  early_stop_length=td_params['early_stop_length'],
                                  early_stop_patience=td_params['early_stop_patience'],
                                  verbose=True,
                                  save_model=False,
                                  save_model_name=None,
                                  auto_load_model=False,
                                  return_outputs=True)

            if dynamic_mode:
                feature_maps = dynamic_fm()

            prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                          period_feature=data_loader.td_loader.test_period,
                                          trend_feature=data_loader.td_loader.test_trend,
                                          laplace_matrix=td_graphBuilder.LM,
                                          target=data_loader.td_loader.test_y,
                                          external_feature=data_loader.td_loader.test_ef,
                                          output_names=('prediction',),
                                          sequence_length=data_loader.td_loader.test_sequence_len,
                                          cache_volume=td_params['batch_size'], )

            transfer_prediction = prediction['prediction']
            test_rmse = metric.rmse(prediction=td_de_normalizer(prediction['prediction']),
                                    target=td_de_normalizer(data_loader.td_loader.test_y))

            validate_error = output[-1]['val_loss'] if validate_mode == 'val' else test_rmse

            if early_stop.stop(validate_error):
                break

            if best_value is None or best_value > validate_error:
                best_value = validate_error
                td_model.save(transfer_model_name, global_step=0)

            print(epoch, 'test rmse', test_rmse)

    td_model.load(transfer_model_name)

    prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                  period_feature=data_loader.td_loader.test_period,
                                  trend_feature=data_loader.td_loader.test_trend,
                                  laplace_matrix=td_graphBuilder.LM,
                                  target=data_loader.td_loader.test_y,
                                  external_feature=data_loader.td_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=data_loader.td_loader.test_sequence_len,
                                  cache_volume=td_params['batch_size'], )

    transfer_prediction = prediction['prediction']

    transfer_error_station = np.array([metric.rmse(td_de_normalizer(transfer_prediction[:, i]),
                                                   td_de_normalizer(data_loader.td_loader.test_y[:, i]))
                                       for i in range(len(transfer_prediction[0]))])

    test_rmse, test_mape = metric.rmse(prediction=td_de_normalizer(transfer_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y)), \
                           metric.mape(prediction=td_de_normalizer(transfer_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y), threshold=0)

    print('#################################################################')
    print('Target Domain Transfer')
    print(test_rmse, test_mape)
    writing_obj.append('%.5f' % test_rmse)

with open('transfer_record.md', 'a+', encoding='utf-8') as f:
    f.write('|' + '|'.join(writing_obj) + '|' + '\n')

# Plot
# data_loader.td_loader.st_map(build_order=finetune_error_station-transfer_error_station)

# for index in range(0, data_loader.td_loader.station_number, 10):
#
#     show_prediction(pretrain=pretrain_prediction,
#                     finetune=finetune_prediction,
#                     transfer=transfer_prediction,
#                     target=data_loader.td_loader.test_y,
#                     station_index=index, start=0, end=500)