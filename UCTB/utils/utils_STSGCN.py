import time
import numpy as np
import mxnet as mx
from UCTB.train.LossFunction import masked_mse_np
from UCTB.evaluation.metric import rmse, mape
from UCTB.preprocess.GraphGenerator import GraphGenerator
from UCTB.preprocess import SplitData
from UCTB.model.STSGCN import construct_model


def configData(args, data_loader, batch_size, config, ctx):
    print("args['normalize']", args.normalize)
    normalizer = data_loader.normalizer
    graph_obj = GraphGenerator(graph='distance', data_loader=data_loader)

    config["num_of_vertices"] = data_loader.station_number
    config["points_per_hour"] = data_loader.closeness_len + data_loader.period_len + data_loader.trend_len
    num_of_vertices = config["num_of_vertices"]
    net = construct_model(config, AM=graph_obj.AM[0])

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

    seq_train = np.concatenate([train_trend, train_period, train_closeness], axis=2).transpose([0, 2, 1, 3])
    seq_val = np.concatenate([val_trend, val_period, val_closeness], axis=2).transpose([0, 2, 1, 3])
    seq_test = np.concatenate([data_loader.test_trend, data_loader.test_period, data_loader.test_closeness],
                                  axis=2).transpose([0, 2, 1, 3])


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
    return model_name, epochs, metric, mod, train_loader, val_loader, test_loader, normalizer, val_y, test_y, all_info


def training(epochs, metric, mod, train_loader, val_loader, test_loader, normalizer, val_y, test_y, all_info):
    global global_epoch
    global_epoch = 1
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

            prediction = normalizer.inverse_transform(prediction)
            de_norm_test_y = normalizer.inverse_transform(test_y)
            rmse_result = rmse(prediction=prediction.squeeze(), target=de_norm_test_y.squeeze())
            mape_result = mape(prediction=prediction.squeeze(), target=de_norm_test_y.squeeze(), threshold=0.01)

            print('test: Epoch: {}, MAPE: {:.2f}, RMSE: {:.2f}, '
                  'time: {:.2f}s'.format(
                global_epoch, mape_result, rmse_result, time.time() - t))
            print(flush=True)
            info.extend((mape_result, rmse_result))
            all_info.append(info)
            lowest_val_loss = loss

        global_epoch += 1



