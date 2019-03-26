## Quick Start with AMulti-GCLSTM

Following shows a quick start of AMulti-GCLSTM using data loader from Dataset. 

```python
from DataSet.node_traffic_loader import NodeTrafficLoader
from Model.AMulti_GCLSTM import AMulti_GCLSTM
from EvalClass.Accuracy import Accuracy

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC')

AMulti_GCLSTM_Obj = AMulti_GCLSTM(T=6, num_node=data_loader.station_number,
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim)

AMulti_GCLSTM_Obj.build()

# Training
AMulti_GCLSTM_Obj.fit(input=data_loader.train_x,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      external_feature=data_loader.train_ef)

# Evaluate
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