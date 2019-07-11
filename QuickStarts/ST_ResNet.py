from UCTB.dataset import NodeTrafficLoader
from UCTB.model import ST_ResNet
from UCTB.evaluation import metric

# Config data loader
data_loader = NodeTrafficLoader(dataset='DiDi', city='Xian', closeness_len=6, period_len=7, trend_len=4)

AMulti_GCLSTM_Obj = ST_ResNet(closeness_len=data_loader.closeness_len,
                              period_len=data_loader.period_len,
                              trend_len=data_loader.trend_len,
                              external_dim=data_loader.external_dim)

AMulti_GCLSTM_Obj.build()

# Training
AMulti_GCLSTM_Obj.fit(closeness_feature=data_loader.train_closeness,
                      period_feature=data_loader.train_period,
                      trend_feature=data_loader.train_trend,
                      laplace_matrix=data_loader.LM,
                      target=data_loader.train_y,
                      external_feature=data_loader.train_ef,
                      sequence_length=data_loader.train_sequence_len,
                      validate_ratio=0.1)

# Predict
prediction = AMulti_GCLSTM_Obj.predict(closeness_feature=data_loader.test_closeness,
                                       period_feature=data_loader.test_period,
                                       trend_feature=data_loader.test_trend,
                                       laplace_matrix=data_loader.LM,
                                       target=data_loader.test_y,
                                       external_feature=data_loader.test_ef,
                                       sequence_length=data_loader.test_sequence_len)

# Compute metric
print('Test result', metric.rmse(prediction=prediction['prediction'], target=data_loader.test_y))