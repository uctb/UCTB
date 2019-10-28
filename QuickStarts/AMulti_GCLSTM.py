from UCTB.dataset import NodeTrafficLoader
from UCTB.model import AMultiGCLSTM
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Metro', city='Shanghai', closeness_len=6, period_len=7, trend_len=4)

AMulti_GCLSTM_Obj = AMultiGCLSTM(closeness_len=data_loader.closeness_len,
                                 period_len=data_loader.period_len,
                                 trend_len=data_loader.trend_len,
                                 num_node=data_loader.station_number,
                                 num_graph=data_loader.LM.shape[0],
                                 external_dim=data_loader.external_dim)

print('Number of trainable variables', AMulti_GCLSTM_Obj.trainable_vars)
print('Number of training samples', data_loader.train_sequence_len)
print('Number of nodes', data_loader.station_number)

# Build tf-graph
AMulti_GCLSTM_Obj.build()

# Training
AMulti_GCLSTM_Obj.fit(closeness_feature=data_loader.train_closeness,
                      period_feature=data_loader.train_period,
                      trend_feature=data_loader.train_trend,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      external_feature=data_loader.train_ef,
                      sequence_length=data_loader.train_sequence_len)

# Predict
prediction = AMulti_GCLSTM_Obj.predict(closeness_feature=data_loader.test_closeness,
                                       period_feature=data_loader.test_period,
                                       trend_feature=data_loader.test_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       output_names=['prediction'],
                                       sequence_length=data_loader.test_sequence_len)

# Evaluate
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0))