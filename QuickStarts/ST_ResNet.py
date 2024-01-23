from UCTB.dataset import GridTrafficLoader
from UCTB.model import ST_ResNet
from UCTB.evaluation import metric

# Config data loader
data_loader = GridTrafficLoader(dataset='Bike', city='Chicago', closeness_len=6, period_len=7, trend_len=4)

ST_ResNet_Obj = ST_ResNet(closeness_len=data_loader.closeness_len,
                          period_len=data_loader.period_len,
                          trend_len=data_loader.trend_len,
                          external_dim=data_loader.external_dim,
                          width=data_loader.width, height=data_loader.height)

ST_ResNet_Obj.build()

print(ST_ResNet_Obj.trainable_vars)

# Training
ST_ResNet_Obj.fit(closeness_feature=data_loader.train_closeness,
                  period_feature=data_loader.train_period,
                  trend_feature=data_loader.train_trend,
                  target=data_loader.train_y,
                  external_feature=data_loader.train_ef,
                  sequence_length=data_loader.train_sequence_len,
                  validate_ratio=0.1)

# Predict
prediction = ST_ResNet_Obj.predict(closeness_feature=data_loader.test_closeness,
                                   period_feature=data_loader.test_period,
                                   trend_feature=data_loader.test_trend,
                                   target=data_loader.test_y,
                                   external_feature=data_loader.test_ef,
                                   sequence_length=data_loader.test_sequence_len)

# Compute metric
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y)))