from UCTB.dataset import NodeTrafficLoader
from UCTB.model import STMeta
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='Bike', city='NYC', graph='Correlation',
                                closeness_len=6, period_len=7, trend_len=4, normalize=True)

# Init model object
STMeta_Obj = STMeta(closeness_len=data_loader.closeness_len,
                                 period_len=data_loader.period_len,
                                 trend_len=data_loader.trend_len,
                                 num_node=data_loader.station_number,
                                 num_graph=data_loader.LM.shape[0])

# Build tf-graph
STMeta_Obj.build()
# Training
STMeta_Obj.fit(closeness_feature=data_loader.train_closeness,
                      period_feature=data_loader.train_period,
                      trend_feature=data_loader.train_trend,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      sequence_length=data_loader.train_sequence_len)

# Predict
prediction = STMeta_Obj.predict(closeness_feature=data_loader.test_closeness,
                                       period_feature=data_loader.test_period,
                                       trend_feature=data_loader.test_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       output_names=['prediction'],
                                       sequence_length=data_loader.test_sequence_len)

# Evaluate
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0))