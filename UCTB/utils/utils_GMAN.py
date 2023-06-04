import time
import math
import os
from UCTB.model.GMAN import GMAN
from gensim.models import Word2Vec
import datetime
from UCTB.preprocess import SplitData
from UCTB.train.LossFunction import mae_loss
import tensorflow as tf
import networkx as nx
import numpy as np


def build_model(log, time_fitness, trainX, args, SE):
    log_string(log, 'compiling model...')
    T = time_fitness
    print("time_fitness: ", T)
    num_train, _, N = trainX.shape
    X, TE, label, is_training = placeholder(args.P, args.Q, N)
    global_step = tf.Variable(0, trainable=False)
    bn_momentum = tf.compat.v1.train.exponential_decay(
        0.5, global_step,
        decay_steps=args.decay_epoch * num_train // args.batch_size,
        decay_rate=0.5, staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)

    pred = GMAN(
        X, TE, SE, args.P, args.Q, T, args.L, args.K, args.d,
        bn=True, bn_decay=bn_decay, is_training=is_training)
    
    loss = mae_loss(pred, label)
    tf.compat.v1.add_to_collection('pred', pred)
    tf.compat.v1.add_to_collection('loss', loss)
    learning_rate = tf.compat.v1.train.exponential_decay(
        args.learning_rate, global_step,
        decay_steps=args.decay_epoch * num_train // args.batch_size,
        decay_rate=0.7, staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-5)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    log_string(log, 'trainable parameters: {:,}'.format(parameters))
    log_string(log, 'model compiled!')
    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    return X, TE, label, is_training, saver, sess, train_op, loss, pred


def Train(log, args, trainX, trainY, trainTE, valX, valTE, valY, X, TE, label, is_training, saver, sess, train_op, loss, pred):

    log_string(log, '**** training model ****')
    num_train, _, N = trainX.shape
    num_val = valX.shape[0]
    wait = 0
    val_loss_min = np.inf
    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, 'early stop at epoch: %04d' % (epoch))
            break
        # shuffle
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        # train loss
        start_train = time.time()
        train_loss = 0
        num_batch = math.ceil(num_train / args.batch_size)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            feed_dict = {
                X: trainX[start_idx: end_idx],
                TE: trainTE[start_idx: end_idx],
                label: trainY[start_idx: end_idx],
                is_training: True}
            # print(train_op.size())
            _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_batch * (end_idx - start_idx)
        train_loss /= num_train
        end_train = time.time()
        # val loss
        start_val = time.time()
        val_loss = 0
        num_batch = math.ceil(num_val / args.batch_size)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            feed_dict = {
                X: valX[start_idx: end_idx],
                TE: valTE[start_idx: end_idx],
                label: valY[start_idx: end_idx],
                is_training: False}
            loss_batch = sess.run(loss, feed_dict=feed_dict)
            val_loss += loss_batch * (end_idx - start_idx)
        val_loss /= num_val
        end_val = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
        if val_loss <= val_loss_min:
            log_string(
                log,
                'val loss decrease from %.4f to %.4f, saving model to %s' %
                (val_loss_min, val_loss, args.model_file))
            wait = 0
            val_loss_min = val_loss
            saver.save(sess, args.model_file)
        else:
            wait += 1
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
    saver.restore(sess, args.model_file)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    trainPred = []
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: trainX[start_idx: end_idx],
            TE: trainTE[start_idx: end_idx],
            is_training: False}
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        trainPred.append(pred_batch)
    trainPred = np.concatenate(trainPred, axis=0)

    valPred = []
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: valX[start_idx: end_idx],
            TE: valTE[start_idx: end_idx],
            is_training: False}
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        valPred.append(pred_batch)
    valPred = np.concatenate(valPred, axis=0)

    return trainPred, valPred


def Test(log, args, testX, testTE, X, TE, is_training, sess, pred):
    num_test = testX.shape[0]
    testPred = []
    num_batch = math.ceil(num_test / args.batch_size)
    start_test = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: testX[start_idx: end_idx],
            TE: testTE[start_idx: end_idx],
            is_training: False}
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        testPred.append(pred_batch)
    end_test = time.time()
    testPred = np.concatenate(testPred, axis=0)

    log_string(log, 'Test time: %.1fmin' % ((end_test - start_test) / 60))
    sess.close()
    log.close()

    return testPred


def load_data(args, data_loader):
    # split data
    train_closeness, val_closeness = SplitData.split_data(
        data_loader.train_closeness, [0.9, 0.1])
    train_period, val_period = SplitData.split_data(
        data_loader.train_period, [0.9, 0.1])
    train_trend, val_trend = SplitData.split_data(
        data_loader.train_trend, [0.9, 0.1])
    train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])
    print(train_y.shape)

    trainY = train_y.reshape([train_y.shape[0], 1, data_loader.station_number])
    valY = val_y.reshape([val_y.shape[0], 1, data_loader.station_number])
    testY = data_loader.test_y.reshape(
        [data_loader.test_y.shape[0], 1, data_loader.station_number])

    # T, num_node, dimension -> T, dimension, num_node
    if data_loader.period_len > 0 and data_loader.trend_len > 0:
        trainX = np.concatenate(
            [train_trend, train_period, train_closeness], axis=2).squeeze().transpose([0, 2, 1])
        print(trainX.shape)
        valX = np.concatenate(
            [val_trend, val_period, val_closeness], axis=2).squeeze().transpose([0, 2, 1])
        testX = np.concatenate([data_loader.test_trend, data_loader.test_period, data_loader.test_closeness],
                               axis=2).squeeze().transpose([0, 2, 1])
    else:
        trainX = train_closeness.squeeze().transpose([0, 2, 1])
        valX = val_closeness.squeeze().transpose([0, 2, 1])
        testX = data_loader.test_closeness.squeeze().transpose([0, 2, 1])


    # spatial embedding
    f = open(args.SE_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    train_steps, val_steps, test_steps = len(trainX), len(valX), len(testX)
    # print("train_steps",train_steps)
    # print("test_steps",test_steps)
    # print("val_steps",val_steps)

    # temporal embedding
    time_fitness = data_loader.dataset.time_fitness
    time_delta = datetime.timedelta(minutes=time_fitness)
    try:
        start_time = datetime.datetime.strptime(
            data_loader.dataset.time_range[0], "%Y-%m-%d")
    except:
        start_time = datetime.datetime.strptime(
            data_loader.dataset.time_range[0], "%Y-%m-%d %H:%M:%S")

    Time = [start_time + i *
            time_delta for i in range(train_steps + test_steps + val_steps + args.P)]

    def get_attr(object, attr):
        return np.array([eval("item.{}".format(attr)) for item in object])

    print(data_loader.dataset.time_range[0])
    dayofweek = np.reshape(get_attr(Time, "weekday()"), newshape=(-1, 1))
    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #             // Time.freq.delta.total_seconds()
    timeofday = (get_attr(Time, "hour") * 3600 + get_attr(Time, "minute") * 60 + get_attr(Time, "second")) \
        // (86400)
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    train = Time[: train_steps + args.P]
    val = Time[train_steps: train_steps + val_steps + args.P]
    test = Time[train_steps + val_steps:]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, time_fitness)


def placeholder(P, Q, N):
    X = tf.compat.v1.placeholder(shape=(None, P, N), dtype=tf.float32)
    TE = tf.compat.v1.placeholder(shape=(None, P + Q, 2), dtype=tf.int32)
    label = tf.compat.v1.placeholder(shape=(None, Q, N), dtype=tf.float32)
    is_training = tf.compat.v1.placeholder(shape=(), dtype=tf.bool)
    return X, TE, label, is_training


def graph_to_adj_files(adjacent_matrix, Adj_file):
    with open(Adj_file, "w") as fp:
        adj_list = []
        print(adjacent_matrix.shape)
        for i in range(adjacent_matrix.shape[0]):
            for j in range(adjacent_matrix.shape[1]):
                adj_list.append("{} {} {:.6f}\n".format(
                    i, j, adjacent_matrix[i, j]))
        fp.writelines(adj_list)


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())
    return G


def learn_embeddings(walks, dimensions, output_file, epochs):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=8, epochs=epochs)
    model.wv.save_word2vec_format(output_file)
    return


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)
