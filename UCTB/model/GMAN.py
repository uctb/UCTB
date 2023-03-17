
import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd
import sys
sys.path.append('/mnt/UCTB_master/')
from UCTB.preprocess import GraphGenerator,SplitData
from UCTB.dataset import NodeTrafficLoader
import datetime
from gensim.models import Word2Vec
import os,math
import time, datetime


import networkx as nx
import random


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			if norm_const == 0:
				print("node:",node)
				print("unnormalized_probs:",unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return

def Train_then_Test(log,time_fitness,trainX,args,std,SE,mean,valX,valTE,valY,testX,testTE,testY,start,trainY,trainTE):
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
    pred = pred * std + mean
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
    # import pdb;pdb.set_trace()
    
    log_string(log, '**** training model ****')
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
            print(X.shape)
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
    num_test = testX.shape[0]
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
    train_mae, train_rmse, train_mape = metric(trainPred, trainY)
    val_mae, val_rmse, val_mape = metric(valPred, valY)
    test_mae, test_rmse, test_mape = metric(testPred, testY)

    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))

    print("UCTB API MAPE:", mape(testPred.squeeze(), testY.squeeze(), threshold=0.01))
    print("UCTB API RMSE:", rmse(testPred.squeeze(), testY.squeeze(), threshold=0))

    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    sess.close()
    log.close()

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]


def conv2d(x, output_dims, kernel_size, stride = [1, 1],
           padding = 'SAME', use_bias = True, activation = tf.nn.relu,
           bn = False, bn_decay = None, is_training = None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape = kernel_shape),
        dtype = tf.float32, trainable = True, name = 'kernel')
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding = padding)
    if use_bias:
        bias = tf.Variable(
            tf.zeros_initializer()(shape = [output_dims]),
            dtype = tf.float32, trainable = True, name = 'bias')
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        if bn:
            x = batch_norm(x, is_training = is_training, bn_decay = bn_decay)
        x = activation(x)
    return x

def batch_norm(x, is_training, bn_decay):
    input_dims = x.get_shape()[-1].value
    moment_dims = list(range(len(x.get_shape()) - 1))
    beta = tf.Variable(
        tf.zeros_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True, name = 'beta')
    gamma = tf.Variable(
        tf.ones_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True, name = 'gamma')
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')

    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay = decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

def dropout(x, drop, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate = drop),
        lambda: x)
    return x


def placeholder(P, Q, N):
    X = tf.compat.v1.placeholder(shape=(None, P, N), dtype=tf.float32)
    TE = tf.compat.v1.placeholder(shape=(None, P + Q, 2), dtype=tf.int32)
    label = tf.compat.v1.placeholder(shape=(None, Q, N), dtype=tf.float32)
    is_training = tf.compat.v1.placeholder(shape=(), dtype=tf.bool)
    return X, TE, label, is_training


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x


def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # temporal embedding
    dayofweek = tf.one_hot(TE[..., 0], depth=7)
    timeofday = tf.one_hot(TE[..., 1], depth=T)
    TE = tf.concat((dayofweek, timeofday), axis=-1)
    TE = tf.expand_dims(TE, axis=2)
    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.add(SE, TE)


def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis=-1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis=-1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape=(num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)
        mask = tf.tile(mask, multiples=(K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype=tf.bool)
        attention = tf.compat.v2.where(
            condition=mask, x=attention, y=-2 ** 15 + 1)
    # softmax
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H


def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=False):
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=mask)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)


def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        STE_P, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def GMAN(X, TE, SE, P, Q, T, L, K, d, bn, bn_decay, is_training):
    '''
    GMAN
    X：       [batch_size, P, N]
    TE：      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE：      [N, K * d]
    P：       number of history steps
    Q：       number of prediction steps
    T：       one day is divided into T steps
    L：       number of STAtt blocks in the encoder/decoder
    K：       number of attention heads
    d：       dimension of each attention head outputs
    return：  [batch_size, Q, N]
    '''
    D = K * d
    # input
    X = tf.expand_dims(X, axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P:]
    # encoder
    for _ in range(L):
        X = STAttBlock(X, STE_P, K, d, bn, bn_decay, is_training)
    # transAtt
    X = transformAttention(
        X, STE_P, STE_Q, K, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.squeeze(X, axis=3)


def mae_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    # mask = tf.compat.v2.where(
    #     condition = tf.math.is_nan(mask), x = 0., y = mask)
    mask = tf.compat.v1.where(
        condition=tf.math.is_nan(mask), x=tf.zeros_like(mask), y=mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    # loss = tf.compat.v2.where(
    #     condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.compat.v1.where(
        condition=tf.math.is_nan(loss), x=tf.zeros_like(mask), y=loss)
    loss = tf.reduce_mean(loss)
    return loss



# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y


def loadData(args):
    # Traffic
    df = pd.read_hdf(args.traffic_file)
    Traffic = df.values
    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

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

    # temporal embedding
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #             // Time.freq.delta.total_seconds()
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (86400)
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)
    # print("trainX",trainX.shape)
    # print("trainY",trainY.shape)
    # print("trainTE",trainTE.shape)
    # print("valTE",valTE.shape)
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


def loadData_cly(args):
    # # Traffic
    # df = pd.read_hdf(args.traffic_file)
    # # print("df",type(df))
    # # print(df.head())
    # # print("df index",type(df.index))
    # Traffic = df.values
    # # train/val/test
    # num_step = df.shape[0]
    # train_steps = round(args.train_ratio * num_step)
    # test_steps = round(args.test_ratio * num_step)
    # val_steps = num_step - train_steps - test_steps
    # train = Traffic[: train_steps]
    # val = Traffic[train_steps : train_steps + val_steps]
    # test = Traffic[-test_steps :]
    # # X, Y
    # trainX, trainY = seq2instance(train, args.P, args.Q)
    # valX, valY = seq2instance(val, args.P, args.Q)
    # testX, testY = seq2instance(test, args.P, args.Q)

    data_loader = NodeTrafficLoader(dataset=args.dataset, city=args.city,
                                    data_range=args.data_range, train_data_length=args.train_data_length,
                                    test_ratio=float(args.test_ratio),
                                    closeness_len=args.closeness_len,
                                    period_len=args.period_len,
                                    trend_len=args.trend_len,
                                    normalize=False,
                                    MergeIndex=args.MergeIndex,
                                    MergeWay=args.MergeWay,
                                    remove=True)

    # split data
    train_closeness, val_closeness = SplitData.split_data(data_loader.train_closeness, [0.9, 0.1])
    train_period, val_period = SplitData.split_data(data_loader.train_period, [0.9, 0.1])
    train_trend, val_trend = SplitData.split_data(data_loader.train_trend, [0.9, 0.1])
    train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])
    print(train_y.shape)

    trainY = train_y.reshape([train_y.shape[0], 1, data_loader.station_number])
    valY = val_y.reshape([val_y.shape[0], 1, data_loader.station_number])
    testY = data_loader.test_y.reshape([data_loader.test_y.shape[0], 1, data_loader.station_number])

    # T, num_node, dimension -> T, dimension, num_node
    if data_loader.period_len > 0 and data_loader.trend_len > 0:
        trainX = np.concatenate([train_trend, train_period, train_closeness], axis=2).squeeze().transpose([0, 2, 1])
        print(trainX.shape)
        valX = np.concatenate([val_trend, val_period, val_closeness], axis=2).squeeze().transpose([0, 2, 1])
        testX = np.concatenate([data_loader.test_trend, data_loader.test_period, data_loader.test_closeness],
                               axis=2).squeeze().transpose([0, 2, 1])
    else:
        trainX = train_closeness.squeeze().transpose([0, 2, 1])
        valX = val_closeness.squeeze().transpose([0, 2, 1])
        testX = data_loader.test_closeness.squeeze().transpose([0, 2, 1])

    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

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
        start_time = datetime.datetime.strptime(data_loader.dataset.time_range[0], "%Y-%m-%d")
    except:
        start_time = datetime.datetime.strptime(data_loader.dataset.time_range[0], "%Y-%m-%d %H:%M:%S")

    Time = [start_time + i * time_delta for i in range(train_steps + test_steps + val_steps + args.P)]

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
    # print("trainX",trainX.shape)
    # print("trainY",trainY.shape)
    # print("valX",valX.shape)
    # print("testX",testX.shape)
    # print("trainTE",trainTE.shape)
    # print("valTE",valTE.shape)
    # print("testTE",testTE.shape)
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std, time_fitness)


def rmse(prediction, target, threshold=None):
    """
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target
            will be removed in computing the rmse
    """
    if threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]


def mape(prediction, target, threshold=0):
    """
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the mape,
            the smallest threshold in mape is zero.
    """
    assert threshold > 0
    return (np.dot((np.abs(prediction - target) / (target + (1 - (target > threshold)))).reshape([1, -1]),
                   target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]

def graph_to_adj_files(adjacent_matrix, Adj_file):
    if not os.path.exists(Adj_file):
        with open(Adj_file, "w") as fp:
            adj_list = []
            for i in range(adjacent_matrix.shape[0]):
                for j in range(adjacent_matrix.shape[1]):
                    adj_list.append("{} {} {:.6f}\n".format(i, j, adjacent_matrix[i, j]))
            fp.writelines(adj_list)


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, dimensions, output_file,epochs):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=8, epochs=epochs)
    model.wv.save_word2vec_format(output_file)
    return