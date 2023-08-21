
import tensorflow as tf
import numpy as np
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

	SE:	 [N, D]
	TE:	 [batch_size, P + Q, 2] (dayofweek, timeofday)
	T:	  num of time steps in one day
	D:	  output dims
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

	X:	  [batch_size, num_step, N, D]
	STE:	[batch_size, num_step, N, D]
	K:	  number of attention heads
	d:	  dimension of each attention outputs
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

	X:	  [batch_size, num_step, N, D]
	STE:	[batch_size, num_step, N, D]
	K:	  number of attention heads
	d:	  dimension of each attention outputs
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

	HS:	 [batch_size, num_step, N, D]
	HT:	 [batch_size, num_step, N, D]
	D:	  output dims
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
	
	X:	  [batch_size, P, N, D]
	STE_P:  [batch_size, P, N, D]
	STE_Q:  [batch_size, Q, N, D]
	K:	  number of attention heads
	d:	  dimension of each attention outputs
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
	"""
	References:
        - `Gman: A graph multi-attention network for traffic prediction.
          <https://ojs.aaai.org/index.php/AAAI/article/view/5477>`_.
        - `A Tensorflow implementation of the GMAN model  (Zhengchuanpan)
          <https://github.com/zhengchuanpan/GMAN>`_.

    Args:
		P(int): Number of history steps.
		Q(int): Number of prediction steps.
		T(int): Number of steps which one day is divided into.
		L(int): Number of STAtt blocks in the encoder/decoder.
		K(int): Number of attention heads.
		d(int): Number of dimension of each attention head outputs.
        X(tf.Tensor): Input traffic data with shape [batch_size, ...]
        TE(tf.Tensor): Temporal embedding [batch_size, ...]
        SE(tf.Tensor): Spatial embedding [batch_size, ...]
		bn(bool): Whether to do batch normalization.
		is_training(bool): Whether to train.
    """
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

