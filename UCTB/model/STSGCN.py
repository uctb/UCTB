import numpy as np
import mxnet as mx
from UCTB.train.LossFunction import huber_loss


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





def stsgcn(data, adj, label,
           input_length, num_of_vertices, num_of_features,
           filter_list, module_type, activation,
           use_mask=True, mask_init_value=None,
           temporal_emb=True, spatial_emb=True,
           prefix="", rho=1, predict_length=12):
    """

    References:
        - `Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting.
          <https://ojs.aaai.org/index.php/AAAI/article/view/5438>`_.
        - `A Mxnet implementation of the stsgcn model  (Davidham3)
          <https://github.com/Davidham3/STSGCN>`_.

    Args:
        data(mxnet.sym): Input data.
        adj(mxnet.sym): Adjacent matrix.
        label(mxnet.sym): Prediction label.
        input_length(int): Length of input data.
        num_of_vertices(int): Number of vertices in the graph.
        num_of_features(int): Number of features of each vertice.
        filter_list(list): Filters.
        module_type(str): Whether sharing weights.
        activation(str): Choose which activate function.
        use_mask(bool): Whether we use mask.
        mask_init_value(int): Initial value of mask.
        temporal_emb(bool): Whether to use temporal embedding.
        spatial_emb(bool): Whether to use spatial embedding.
        prefix(str): String prefix of mask.
        rho(float): Hyperparameters used to calculate huber loss.
        predict_length(int): Length of prediction.
    """
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



def construct_model(config, AM):

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



