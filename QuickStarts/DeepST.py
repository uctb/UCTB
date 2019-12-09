from UCTB.dataset import GridTrafficLoader
from UCTB.model import DeepST
from UCTB.evaluation import metric

# Config data loader
data_loader = GridTrafficLoader(dataset='Bike', city='NYC', closeness_len=6, period_len=7, trend_len=4)

deep_st_obj = DeepST(closeness_len=data_loader.closeness_len,
                     period_len=data_loader.period_len,
                     trend_len=data_loader.trend_len,
                     external_dim=data_loader.external_dim,
                     width=data_loader.width, height=data_loader.height,
                     lr=1e-5)

deep_st_obj.build()

print('Trainable variables', deep_st_obj.trainable_vars)

# Training
deep_st_obj.fit(closeness_feature=data_loader.train_closeness,
                period_feature=data_loader.train_period,
                trend_feature=data_loader.train_trend,
                target=data_loader.train_y,
                external_feature=data_loader.train_ef,
                sequence_length=data_loader.train_sequence_len,
                validate_ratio=0.1)

# Predict
prediction = deep_st_obj.predict(closeness_feature=data_loader.test_closeness,
                                 period_feature=data_loader.test_period,
                                 trend_feature=data_loader.test_trend,
                                 target=data_loader.test_y,
                                 external_feature=data_loader.test_ef,
                                 sequence_length=data_loader.test_sequence_len)

# Compute metric
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0))
