## Quick Start with AMulti-GCLSTM

To use this quick start code, you need to first implement a class `data_loader` with int `node_num`, int `T` to represent the previous time series data size, ndarray `graphs` with shape `graph_num * node_num * node_num` to represent Graph Laplace Matrix, and a method `get_batch()` to generate mini batch `(X, y)` with `X`'s shape `batch_size * T * node_num * 1` and `y`'s shape `batch_size * node_num`.

```python
from Model.AMulti_GCLSTM import AMulti_GCLSTM

AMulti_GCLSTM_Obj = AMulti_GCLSTM(num_node=data_loader.node_num,
                                  GCN_K=1, GCN_layers=2,
                                  num_graph=data_loader.graphs.shape[0],
                                  external_dim=None,
                                  GCLSTM_layers=2,
                                  gal_units=256, gal_num_heads=2,
                                  T=data_loader.T, num_filter_conv1x1=32,
                                  num_hidden_units=256, lr=lr,
                                  code_version='AMulti_GCLSTM_0', GPU_DEVICE='0',
                                  model_dir='tf_model_dir')
AMulti_GCLSTM_Obj.build()

for batch in range(batch_num):
    X, y = data_loader.get_batch()
    l = AMulti_GCLSTM_Obj.fit(
        {'input': X,
        'target': y,
        'laplace_matrix': data_loader.graphs,
        'external_input': None},
        global_step=batch, summary=False)['loss']
```

------

<u>[Back To HomePage](../index.html)</u>