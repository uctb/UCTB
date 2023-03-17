# -*- coding:utf-8 -*-

import os
import time
import numpy as np
import mxnet as mx
from UCTB.evaluation.metric import rmse, mape
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.preprocess import SplitData
from UCTB.dataset import NodeTrafficLoader
def configMix(args,data_loader,batch_size,config,ctx):
    print("args['normalize']", args.normalize)
    de_normalizer = None if args.normalize is False else data_loader.normalizer.min_max_denormal
    graph_obj = GraphGenerator(graph='distance', data_loader=data_loader)

    config["num_of_vertices"] = data_loader.station_number
    config["points_per_hour"] = data_loader.closeness_len + data_loader.period_len + data_loader.trend_len
    num_of_vertices = config["num_of_vertices"]
    net = construct_model_cly(config, AM=graph_obj.AM[0])

    model_name = "{}_{}_{}".format(args.dataset, args.city, args.MergeIndex)
    print("model_name:", model_name)

    train_closeness, val_closeness = SplitData.split_data(data_loader.train_closeness, [0.9, 0.1])
    train_period, val_period = SplitData.split_data(data_loader.train_period, [0.9, 0.1])
    train_trend, val_trend = SplitData.split_data(data_loader.train_trend, [0.9, 0.1])
    train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

    # T, num_node, 1 -> T, 1, num_node
    train_y = train_y.transpose([0, 2, 1])
    val_y = val_y.transpose([0, 2, 1])
    test_y = data_loader.test_y.transpose([0, 2, 1])

    # T, num_node, dimension, 1 -> T, dimension, num_node, 1
    if data_loader.period_len > 0 and data_loader.trend_len > 0:
        seq_train = np.concatenate([train_trend, train_period, train_closeness], axis=2).transpose([0, 2, 1, 3])
        seq_val = np.concatenate([val_trend, val_period, val_closeness], axis=2).transpose([0, 2, 1, 3])
        seq_test = np.concatenate([data_loader.test_trend, data_loader.test_period, data_loader.test_closeness],
                                  axis=2).transpose([0, 2, 1, 3])
    else:
        seq_train = train_closeness.transpose([0, 2, 1, 3])
        seq_val = val_closeness.transpose([0, 2, 1, 3])
        seq_test = data_loader.test_closeness.transpose([0, 2, 1, 3])

    print(seq_train.shape, seq_val.shape, seq_test.shape)

    loaders = []
    true_values = []
    for item in ["train", "val", "test"]:
        loaders.append(
            mx.io.NDArrayIter(
                eval("seq_{}".format(item)), eval("{}_y".format(item)),
                batch_size=batch_size,
                shuffle=True,
                label_name='label'
            )
        )
        true_values.append(eval("{}_y".format(item)))

    train_loader, val_loader, test_loader = loaders
    _, val_y, test_y = true_values

    global_epoch = 1
    training_samples = len(seq_train)
    global_train_steps = training_samples // batch_size + 1
    all_info = []
    epochs = config['epochs']

    mod = mx.mod.Module(
        net,
        data_names=['data'],
        label_names=['label'],
        context=ctx
    )

    mod.bind(
        data_shapes=[(
            'data',
            (batch_size, config['points_per_hour'], num_of_vertices, 1)
        ), ],
        label_shapes=[(
            'label',
            (batch_size, config['num_for_predict'], num_of_vertices)
        )]
    )

    mod.init_params(initializer=mx.init.Xavier(magnitude=0.0003))
    lr_sch = mx.lr_scheduler.PolyScheduler(
        max_update=global_train_steps * epochs * config['max_update_factor'],
        base_lr=config['learning_rate'],
        pwr=2,
        warmup_steps=global_train_steps
    )

    mod.init_optimizer(
        optimizer=config['optimizer'],
        optimizer_params=(('lr_scheduler', lr_sch),)
    )

    num_of_parameters = 0
    for param_name, param_value in mod.get_params()[0].items():
        # print(param_name, param_value.shape)
        num_of_parameters += np.prod(param_value.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    metric = mx.metric.create(['RMSE', 'MAE'], output_names=['pred_output'])

    if args.plot:
        graph = mx.viz.plot_network(net)
        graph.format = 'png'
        graph.render('graph')
    return model_name,epochs,metric,mod,train_loader, val_loader, test_loader,de_normalizer,val_y,test_y,all_info


def training(epochs,metric,mod,train_loader,val_loader, test_loader,de_normalizer,val_y,test_y,all_info):
    
    global global_epoch
    global_epoch=1
    lowest_val_loss = np.inf
    for _ in range(epochs):
        t = time.time()
        info = [global_epoch]
        train_loader.reset()
        metric.reset()
        for idx, databatch in enumerate(train_loader):
            # print(databatch,type(databatch))
            mod.forward_backward(databatch)
            mod.update_metric(metric, databatch.label)
            mod.update()
        metric_values = dict(zip(*metric.get()))

        print('training: Epoch: %s, RMSE: %.2f, MAE: %.2f, time: %.2f s' % (
            global_epoch, metric_values['rmse'], metric_values['mae'],
            time.time() - t), flush=True)
        # info.append(metric_values['mae'])
        info.append(metric_values['rmse'])

        val_loader.reset()
        prediction = mod.predict(val_loader)[1].asnumpy()
        # loss = masked_mae_np(val_y, prediction, 0)
        loss = masked_mse_np(val_y, prediction, 0)
        print('validation: Epoch: %s, loss: %.2f, time: %.2f s' % (
            global_epoch, loss, time.time() - t), flush=True)
        info.append(loss)

        if loss < lowest_val_loss:

            test_loader.reset()
            prediction = mod.predict(test_loader)[1].asnumpy()
            if de_normalizer:
                prediction = de_normalizer(prediction)
                de_norm_test_y = de_normalizer(test_y)
            rmse_result = rmse(prediction=prediction.squeeze(), target=test_y.squeeze(), threshold=0)
            mape_result = mape(prediction=prediction.squeeze(), target=test_y.squeeze(), threshold=0.01)

            print('test: Epoch: {}, MAPE: {:.2f}, RMSE: {:.2f}, '
                  'time: {:.2f}s'.format(
                global_epoch, mape_result, rmse_result, time.time() - t))
            print(flush=True)
            info.extend((mape_result, rmse_result))
            all_info.append(info)
            lowest_val_loss = loss

        global_epoch += 1




def position_embedding(data,
                       input_length, num_of_vertices, embedding_size,
                       temporal=True, spatial=True,
                       init=mx.init.Xavier(magnitude=0.0003), prefix=""):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    input_length: int, length of time series, T

    num_of_vertices: int, N

    embedding_size: int, C

    temporal, spatial: bool, whether equip this type of embeddings

    init: mx.initializer.Initializer

    prefix: str

    Returns
    ----------
    data: output shape is (B, T, N, C)
    '''

    temporal_emb = None
    spatial_emb = None

    if temporal:
        # shape is (1, T, 1, C)
        temporal_emb = mx.sym.var(
            "{}_t_emb".format(prefix),
            shape=(1, input_length, 1, embedding_size),
            init=init
        )
    if spatial:
        # shape is (1, 1, N, C)
        spatial_emb = mx.sym.var(
            "{}_v_emb".format(prefix),
            shape=(1, 1, num_of_vertices, embedding_size),
            init=init
        )

    if temporal_emb is not None:
        data = mx.sym.broadcast_add(data, temporal_emb)
    if spatial_emb is not None:
        data = mx.sym.broadcast_add(data, spatial_emb)

    return data


def gcn_operation(data, adj,
                  num_of_filter, num_of_features, num_of_vertices,
                  activation, prefix=""):
    '''
    graph convolutional operation, a simple GCN we defined in paper

    Parameters
    ----------
    data: mx.sym.var, shape is (3N, B, C)

    adj: mx.sym.var, shape is (3N, 3N)

    num_of_filter: int, C'

    num_of_features: int, C

    num_of_vertices: int, N

    activation: str, {'GLU', 'relu'}

    prefix: str

    Returns
    ----------
    output shape is (3N, B, C')

    '''

    assert activation in {'GLU', 'relu'}

    # shape is (3N, B, C)
    data = mx.sym.dot(adj, data)

    if activation == 'GLU':

        # shape is (3N, B, 2C')
        data = mx.sym.FullyConnected(
            data,
            flatten=False,
            num_hidden=2 * num_of_filter
        )

        # shape is (3N, B, C'), (3N, B, C')
        lhs, rhs = mx.sym.split(data, num_outputs=2, axis=2)

        # shape is (3N, B, C')
        return lhs * mx.sym.sigmoid(rhs)

    elif activation == 'relu':

        # shape is (3N, B, C')
        return mx.sym.Activation(
            mx.sym.FullyConnected(
                data,
                flatten=False,
                num_hidden=num_of_filter
            ), activation
        )


def stsgcm(data, adj,
           filters, num_of_features, num_of_vertices,
           activation, prefix=""):
    '''
    STSGCM, multiple stacked gcn layers with cropping and max operation

    Parameters
    ----------
    data: mx.sym.var, shape is (3N, B, C)

    adj: mx.sym.var, shape is (3N, 3N)

    filters: list[int], list of C'

    num_of_features: int, C

    num_of_vertices: int, N

    activation: str, {'GLU', 'relu'}

    prefix: str

    Returns
    ----------
    output shape is (N, B, C')

    '''
    need_concat = []

    for i in range(len(filters)):
        data = gcn_operation(
            data, adj,
            filters[i], num_of_features, num_of_vertices,
            activation=activation,
            prefix="{}_gcn_{}".format(prefix, i)
        )
        need_concat.append(data)
        num_of_features = filters[i]

    # shape of each element is (1, N, B, C')
    need_concat = [
        mx.sym.expand_dims(
            mx.sym.slice(
                i,
                begin=(num_of_vertices, None, None),
                end=(2 * num_of_vertices, None, None)
            ), 0
        ) for i in need_concat
    ]

    # shape is (N, B, C')
    return mx.sym.max(mx.sym.concat(*need_concat, dim=0), axis=0)


def stsgcl(data, adj,
           T, num_of_vertices, num_of_features, filters,
           module_type, activation, temporal_emb=True, spatial_emb=True,
           prefix=""):
    '''
    STSGCL

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    adj: mx.sym.var, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    module_type: str, {'sharing', 'individual'}

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    assert module_type in {'sharing', 'individual'}

    if module_type == 'individual':
        return sthgcn_layer_individual(
            data, adj,
            T, num_of_vertices, num_of_features, filters,
            activation, temporal_emb, spatial_emb, prefix
        )
    else:
        return sthgcn_layer_sharing(
            data, adj,
            T, num_of_vertices, num_of_features, filters,
            activation, temporal_emb, spatial_emb, prefix
        )


def sthgcn_layer_individual(data, adj,
                            T, num_of_vertices, num_of_features, filters,
                            activation, temporal_emb=True, spatial_emb=True,
                            prefix=""):
    '''
    STSGCL, multiple individual STSGCMs

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    adj: mx.sym.var, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    # shape is (B, T, N, C)
    data = position_embedding(data, T, num_of_vertices, num_of_features,
                              temporal_emb, spatial_emb,
                              prefix="{}_emb".format(prefix))
    need_concat = []
    for i in range(T - 2):

        # shape is (B, 3, N, C)
        t = mx.sym.slice(data, begin=(None, i, None, None),
                         end=(None, i + 3, None, None))

        # shape is (B, 3N, C)
        t = mx.sym.reshape(t, (-1, 3 * num_of_vertices, num_of_features))

        # shape is (3N, B, C)
        t = mx.sym.transpose(t, (1, 0, 2))

        # shape is (N, B, C')
        t = stsgcm(
            t, adj, filters, num_of_features, num_of_vertices,
            activation=activation,
            prefix="{}_stsgcm_{}".format(prefix, i)
        )

        # shape is (B, N, C')
        t = mx.sym.swapaxes(t, 0, 1)

        # shape is (B, 1, N, C')
        need_concat.append(mx.sym.expand_dims(t, axis=1))

    # shape is (B, T-2, N, C')
    return mx.sym.concat(*need_concat, dim=1)


def sthgcn_layer_sharing(data, adj,
                         T, num_of_vertices, num_of_features, filters,
                         activation, temporal_emb=True, spatial_emb=True,
                         prefix=""):
    '''
    STSGCL, multiple a sharing STSGCM

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    adj: mx.sym.var, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    # shape is (B, T, N, C)
    data = position_embedding(data, T, num_of_vertices, num_of_features,
                              temporal_emb, spatial_emb,
                              prefix="{}_emb".format(prefix))
    need_concat = []
    for i in range(T - 2):
        # shape is (B, 3, N, C)
        t = mx.sym.slice(data, begin=(None, i, None, None),
                         end=(None, i + 3, None, None))

        # shape is (B, 3N, C)
        t = mx.sym.reshape(t, (-1, 3 * num_of_vertices, num_of_features))

        # shape is (3N, B, C)
        t = mx.sym.swapaxes(t, 0, 1)
        need_concat.append(t)

    # shape is (3N, (T-2)*B, C)
    t = mx.sym.concat(*need_concat, dim=1)

    # shape is (N, (T-2)*B, C')
    t = stsgcm(
        t, adj, filters, num_of_features, num_of_vertices,
        activation=activation,
        prefix="{}_stsgcm".format(prefix)
    )

    # shape is (N, T - 2, B, C)
    t = t.reshape((num_of_vertices, T - 2, -1, filters[-1]))

    # shape is (B, T - 2, N, C)
    return mx.sym.swapaxes(t, 0, 2)


def output_layer(data, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=12):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    num_of_vertices: int, N

    input_length: int, length of time series, T

    num_of_features: int, C

    num_of_filters: int, C'

    predict_length: int, length of predicted time series, T'

    Returns
    ----------
    output shape is (B, T', N)
    '''

    # data shape is (B, N, T, C)
    data = mx.sym.swapaxes(data, 1, 2)

    # (B, N, T * C)
    data = mx.sym.reshape(
        data, (-1, num_of_vertices, input_length * num_of_features)
    )

    # (B, N, C')
    data = mx.sym.Activation(
        mx.sym.FullyConnected(
            data,
            flatten=False,
            num_hidden=num_of_filters
        ), 'relu'
    )

    # (B, N, T')
    data = mx.sym.FullyConnected(
        data,
        flatten=False,
        num_hidden=predict_length
    )

    # (B, T', N)
    data = mx.sym.swapaxes(data, 1, 2)

    return data


def huber_loss(data, label, rho=1):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T', N)

    label: mx.sym.var, shape is (B, T', N)

    rho: float

    Returns
    ----------
    loss: mx.sym
    '''

    loss = mx.sym.abs(data - label)
    loss = mx.sym.where(loss > rho, loss - 0.5 * rho,
                        (0.5 / rho) * mx.sym.square(loss))
    loss = mx.sym.MakeLoss(loss)
    return loss


def weighted_loss(data, label, input_length, rho=1):
    '''
    weighted loss build on huber loss

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T', N)

    label: mx.sym.var, shape is (B, T', N)

    input_length: int, T'

    rho: float

    Returns
    ----------
    agg_loss: mx.sym
    '''

    # shape is (1, T, 1)
    weight = mx.sym.expand_dims(
        mx.sym.expand_dims(
            mx.sym.flip(mx.sym.arange(1, input_length + 1), axis=0),
            axis=0
        ), axis=-1
    )
    agg_loss = mx.sym.broadcast_mul(
        huber_loss(data, label, rho),
        weight
    )
    return agg_loss


def stsgcn(data, adj, label,
           input_length, num_of_vertices, num_of_features,
           filter_list, module_type, activation,
           use_mask=True, mask_init_value=None,
           temporal_emb=True, spatial_emb=True,
           prefix="", rho=1, predict_length=12):
    '''
    data shape is (B, T, N, C)
    adj shape is (3N, 3N)
    label shape is (B, T, N)
    '''
    if use_mask:
        if mask_init_value is None:
            raise ValueError("mask init value is None!")
        mask = mx.sym.var("{}_mask".format(prefix),
                          shape=(3 * num_of_vertices, 3 * num_of_vertices),
                          init=mask_init_value)
        adj = mask * adj

    for idx, filters in enumerate(filter_list):
        data = stsgcl(data, adj, input_length, num_of_vertices,
                      num_of_features, filters, module_type,
                      activation=activation,
                      temporal_emb=temporal_emb,
                      spatial_emb=spatial_emb,
                      prefix="{}_stsgcl_{}".format(prefix, idx))
        input_length -= 2
        num_of_features = filters[-1]

    # (B, 1, N)
    need_concat = []
    for i in range(predict_length):
        need_concat.append(
            output_layer(
                data, num_of_vertices, input_length, num_of_features,
                num_of_filters=128, predict_length=1
            )
        )
    data = mx.sym.concat(*need_concat, dim=1)

    loss = huber_loss(data, label, rho=rho)
    return mx.sym.Group([loss, mx.sym.BlockGrad(data, name='pred')])



def construct_model(config):

    module_type = config['module_type']
    act_type = config['act_type']
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']
    batch_size = config['batch_size']

    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    points_per_hour = config['points_per_hour']
    num_for_predict = config['num_for_predict']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    adj = get_adjacency_matrix(adj_filename, num_of_vertices,
                               id_filename=id_filename)
    print("Adj:",adj.shape)
    adj_mx = construct_adj(adj, 3)
    print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)

    data = mx.sym.var("data")
    label = mx.sym.var("label")
    adj = mx.sym.Variable('adj', shape=adj_mx.shape,
                          init=mx.init.Constant(value=adj_mx.tolist()))
    adj = mx.sym.BlockGrad(adj)
    mask_init_value = mx.init.Constant(value=(adj_mx != 0)
                                       .astype('float32').tolist())

    filters = config['filters']
    first_layer_embedding_size = config['first_layer_embedding_size']
    if first_layer_embedding_size:
        data = mx.sym.Activation(
            mx.sym.FullyConnected(
                data,
                flatten=False,
                num_hidden=first_layer_embedding_size
            ),
            act_type='relu'
        )
    else:
        first_layer_embedding_size = num_of_features
    net = stsgcn(
        data, adj, label,
        points_per_hour, num_of_vertices, first_layer_embedding_size,
        filters, module_type, act_type,
        use_mask, mask_init_value, temporal_emb, spatial_emb,
        prefix="", rho=1, predict_length=12
    )
    assert net.infer_shape(
        data=(batch_size, points_per_hour, num_of_vertices, 1),
        label=(batch_size, num_for_predict, num_of_vertices)
    )[1][1] == (batch_size, num_for_predict, num_of_vertices)
    return net


def construct_model_cly(config, AM):

    module_type = config['module_type']
    act_type = config['act_type']
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']
    batch_size = config['batch_size']

    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    points_per_hour = config['points_per_hour']
    num_for_predict = config['num_for_predict']

    adj = AM
    # print("Adj:",adj.shape,adj)
    adj_mx = construct_adj(adj, 3)
    print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)

    data = mx.sym.var("data")
    label = mx.sym.var("label")
    adj = mx.sym.Variable('adj', shape=adj_mx.shape,
                          init=mx.init.Constant(value=adj_mx.tolist()))
    adj = mx.sym.BlockGrad(adj)
    mask_init_value = mx.init.Constant(value=(adj_mx != 0)
                                       .astype('float32').tolist())

    filters = config['filters']
    first_layer_embedding_size = config['first_layer_embedding_size']
    if first_layer_embedding_size:
        data = mx.sym.Activation(
            mx.sym.FullyConnected(
                data,
                flatten=False,
                num_hidden=first_layer_embedding_size
            ),
            act_type='relu'
        )
    else:
        first_layer_embedding_size = num_of_features
    net = stsgcn(
        data, adj, label,
        points_per_hour, num_of_vertices, first_layer_embedding_size,
        filters, module_type, act_type,
        use_mask, mask_init_value, temporal_emb, spatial_emb,
        prefix="", rho=1, predict_length=1
    )

    assert net.infer_shape(
        data=(batch_size, points_per_hour, num_of_vertices, 1),
        label=(batch_size, num_for_predict, num_of_vertices)
    )[1][1] == (batch_size, num_for_predict, num_of_vertices)
    return net

def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def construct_adj(A, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_data(graph_signal_matrix_filename, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
