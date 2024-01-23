import os
import yaml
import argparse
import GPUtil
import numpy as np

from UCTB.dataset import TransferDataLoader
from UCTB.model import STMeta
from UCTB.evaluation import metric

#####################################################################
# argument parser
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('-m', '--model', default='STMeta_v4.model.yml')
parser.add_argument('-sd', '--source_data', default='bike_dc.data.yml')
parser.add_argument('-td', '--target_data', default='bike_dc.data.yml')
parser.add_argument('-tdl', '--target_data_length', default='365', type=str)
parser.add_argument('-pt', '--pretrain', default='True')
parser.add_argument('-ft', '--finetune', default='True')
parser.add_argument('-tr', '--transfer', default='True')

args = vars(parser.parse_args())

with open(args['model'], 'r') as f:
    model_params = yaml.load(f)

with open(args['source_data'], 'r') as f:
    sd_params = yaml.load(f)

with open(args['target_data'], 'r') as f:
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

sub_code_version = 'C{}P{}T{}_G{}_TP'.format(sd_params['closeness_len'], sd_params['period_len'], sd_params['trend_len'],
                                             ''.join([e[0].upper() for e in sd_params['graph'].split('-')]))

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')
model_dir_path = os.path.join(model_dir_path, group)
#####################################################################
# Config data loader

data_loader = TransferDataLoader(sd_params, td_params, model_params, td_data_length=args['target_data_length'])

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=1,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    current_device = str(deviceIDs[0])

sd_model = STMeta(num_node=data_loader.sd_loader.station_number,
                  num_graph=data_loader.sd_loader.LM.shape[0],
                  external_dim=data_loader.sd_loader.external_dim,
                  tpe_dim=data_loader.sd_loader.tpe_dim,
                  code_version=code_version,
                  model_dir=model_dir_path,
                  gpu_device=current_device,
                  transfer_ratio=0,
                  **sd_params, **model_params)
sd_model.build()

td_model = STMeta(num_node=data_loader.td_loader.station_number,
                  num_graph=data_loader.td_loader.LM.shape[0],
                  external_dim=data_loader.td_loader.external_dim,
                  tpe_dim=data_loader.td_loader.tpe_dim,
                  code_version=code_version,
                  model_dir=model_dir_path,
                  transfer_ratio=0.1,
                  gpu_device=current_device,
                  **td_params, **model_params)
td_model.build()

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
transfer_model_name = 'Transfer_' + sub_code_version + '_' + str(data_loader.td_loader.train_sequence_len)

if args['pretrain'] == 'True':

    try:
        td_model.load(pretrain_model_name)

    except FileNotFoundError:

        traffic_sim = data_loader.traffic_sim_fake()

        # prepare data:
        feature_maps = []
        for record in traffic_sim:
            # score, index, start, end
            # sd_transfer_data = data_loader.sd_loader.train_data[record[2]: record[3], :]
            sd_transfer_data = data_loader.sd_loader.train_data[-data_loader.td_loader.train_data.shape[0]:, :]

            transfer_closeness, \
            transfer_period, \
            transfer_trend, \
            _ = data_loader.sd_loader.st_move_sample.move_sample(sd_transfer_data)

            fm = sd_model.predict(closeness_feature=transfer_closeness,
                                  period_feature=transfer_period,
                                  trend_feature=transfer_trend,
                                  laplace_matrix=data_loader.sd_loader.LM,
                                  external_feature=data_loader.sd_loader.train_ef,
                                  output_names=['feature_map'],
                                  sequence_length=len(transfer_closeness),
                                  cache_volume=sd_params['batch_size'])

            feature_maps.append(fm['feature_map'][:, record[1]:record[1] + 1, :, :])

        feature_maps = np.concatenate(feature_maps, axis=1)

        # transfer
        td_model.fit(closeness_feature=data_loader.td_loader.train_closeness,
                     period_feature=data_loader.td_loader.train_period,
                     trend_feature=data_loader.td_loader.train_trend,
                     laplace_matrix=data_loader.td_loader.LM,
                     target=data_loader.td_loader.train_y,
                     external_feature=data_loader.td_loader.train_ef,
                     similar_feature_map=feature_maps,
                     sequence_length=data_loader.td_loader.train_sequence_len,
                     output_names=('transfer_loss',),
                     evaluate_loss_name='transfer_loss',
                     op_names=('transfer_op',),
                     batch_size=td_params['batch_size'],
                     max_epoch=td_params['max_epoch'],
                     validate_ratio=0.1,
                     early_stop_method='t-test',
                     early_stop_length=td_params['early_stop_length'],
                     early_stop_patience=td_params['early_stop_patience'],
                     verbose=True,
                     save_model=True)
        td_model.save(pretrain_model_name, global_step=0)

    prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                  period_feature=data_loader.td_loader.test_period,
                                  trend_feature=data_loader.td_loader.test_trend,
                                  laplace_matrix=data_loader.td_loader.LM,
                                  target=data_loader.td_loader.test_y,
                                  external_feature=data_loader.td_loader.test_ef,
                                  output_names=('prediction',),
                                  sequence_length=data_loader.td_loader.test_sequence_len,
                                  cache_volume=td_params['batch_size'], )

    transfer_prediction = prediction['prediction']

    test_rmse, test_mape = metric.rmse(prediction=td_de_normalizer(transfer_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y)), \
                           metric.mape(prediction=td_de_normalizer(transfer_prediction),
                                       target=td_de_normalizer(data_loader.td_loader.test_y), threshold=0)

    print('#################################################################')
    print('Target Domain Transfer')
    print(test_rmse, test_mape)