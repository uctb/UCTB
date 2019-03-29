from UCTB.dataset import NodeTrafficLoader
from UCTB.model import AMulti_GCLSTM
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='ChargeStation', city='Beijing')

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
                                       metrics=[metric.rmse],
                                       threshold=0)

print('Test result', test_rmse)