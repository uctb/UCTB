
import os
from time import time
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from UCTB.preprocess import SplitData
# from tensorboardX import SummaryWriter

from UCTB.train.LossFunction import masked_mape, masked_mae, masked_rmse, masked_mse


def load_data(data_loader, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''
    # split data
    train_closeness, val_closeness = SplitData.split_data(
        data_loader.train_closeness, [0.9, 0.1])
    train_period, val_period = SplitData.split_data(
        data_loader.train_period, [0.9, 0.1])
    train_trend, val_trend = SplitData.split_data(
        data_loader.train_trend, [0.9, 0.1])
    train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

    train_x = np.concatenate(
        [train_trend, train_period, train_closeness], axis=2).transpose([0, 1, 3, 2])
    train_target = train_y

    val_x = np.concatenate(
        [val_trend, val_period, val_closeness], axis=2).transpose([0, 1, 3, 2])
    val_target = val_y

    test_x = np.concatenate([data_loader.test_trend, data_loader.test_period,
                            data_loader.test_closeness], axis=2).transpose([0, 1, 3, 2])
    test_target = data_loader.test_y



    print("train_x", train_x.shape)
    print("val_x", val_x.shape)
    print("test_x", test_x.shape)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(
        train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(
        val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(
        test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor


def train_main(training_config, params_path, DEVICE, net, val_loader, train_loader, graph_signal_matrix_filename):
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    start_epoch = int(training_config['start_epoch'])
    batch_size = int(training_config['batch_size'])
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    time_strides = num_of_hours
    nb_chev_filter = int(training_config['nb_chev_filter'])
    nb_time_filter = int(training_config['nb_time_filter'])
    in_channels = int(training_config['in_channels'])
    nb_block = int(training_config['nb_block'])
    K = int(training_config['K'])
    loss_function = training_config['loss_function']
    metric_method = training_config['metric_method']
    missing_value = float(training_config['missing_value'])
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse  # nn.MSELoss().to(DEVICE)
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(
            params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(
                net, val_loader, criterion_masked, masked_flag, missing_value, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(
                net, val_loader, criterion, masked_flag, missing_value, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            # sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 10 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' %
                      (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)
    return best_epoch


def predict_main(net, global_step, data_loader, data_target_tensor, params_path):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(
        params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            # (batch, T', 1)
            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' %
                      (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        return prediction
        


def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        # sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

