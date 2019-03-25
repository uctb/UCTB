<link rel="stylesheet" href="./static/css/misty.css" type="text/css"/>

## Quick Start with AMulti-GCLSTM

Following shows a quick start of AMulti-GCLSTM using data loader from Dataset. 

```python
from DataSet.node_traffic_loader import NodeTrafficLoader
from Model.AMulti_GCLSTM import AMulti_GCLSTM
from EvalClass.Accuracy import Accuracy

# Get data loader
data_loader = NodeTrafficLoader()

# Initialize an object of Amulti-GCLSTM
AMulti_GCLSTM_Obj = AMulti_GCLSTM(num_node=data_loader.station_number,
                                  GCN_K=[1],
                                  GCN_layers=[1],
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim,
                                  GCLSTM_layers=1,
                                  gal_units=32,
                                  gal_num_heads=2,
                                  T=6,
                                  num_filter_conv1x1=32,
                                  num_hidden_units=32,
                                  lr=5e-4,
                                  code_version='DebugV0',
                                  GPU_DEVICE='0',
                                  model_dir='./model_dir/')
# Build graphs
AMulti_GCLSTM_Obj.build()

# Train the model
AMulti_GCLSTM_Obj.fit(input=data_loader.train_x,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      external_feature=data_loader.train_ef,
                      batch_size=64,
                      max_epoch=5000,
                      early_stop_method='t-test',
                      early_stop_length=50,
                      early_stop_patience=0.1)

# Evaluate on test data
test_rmse = AMulti_GCLSTM_Obj.evaluate(input=data_loader.test_x,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       metrics=[Accuracy.RMSE],
                                       threshold=0)
print('Test result', test_rmse)
```

------

<u>[Back To HomePage](../index.html)</u>