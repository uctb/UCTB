import nni

from UCTB.dataset import GridTrafficLoader
from UCTB.model import ST_ResNet
from UCTB.evaluation import metric

args = {
    'dataset': 'DiDi',
    'city': 'Xian',
    'num_residual_unit': 4,
    'conv_filters': 64,
    'kernal_size': 3,
    'lr': 5e-5
}

code_version = 'ST_ResNet_{}_{}'.format(args['dataset'], args['city'])

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    code_version += ('_' + str(nni_sid))

# Config data loader
data_loader = GridTrafficLoader(dataset=args['dataset'], city=args['city'], closeness_len=6, period_len=7, trend_len=4)

ST_ResNet_Obj = ST_ResNet(closeness_len=data_loader.closeness_len,
                          period_len=data_loader.period_len,
                          trend_len=data_loader.trend_len,
                          external_dim=data_loader.external_dim,
                          num_residual_unit=args['num_residual_unit'], conv_filters=args['conv_filters'],
                          kernal_size=args['kernal_size'], width=data_loader.width, height=data_loader.height,
                          code_version=code_version)

ST_ResNet_Obj.build()

print(args['dataset'], args['city'], code_version)
print('Number of trainable variables', ST_ResNet_Obj.trainable_vars)
print('Number of training samples', data_loader.train_sequence_len)

# Training
ST_ResNet_Obj.fit(closeness_feature=data_loader.train_closeness,
                  period_feature=data_loader.train_period,
                  trend_feature=data_loader.train_trend,
                  target=data_loader.train_y,
                  external_feature=data_loader.train_ef,
                  sequence_length=data_loader.train_sequence_len,
                  early_stop_length=100,
                  validate_ratio=0.1)

# Predict
prediction = ST_ResNet_Obj.predict(closeness_feature=data_loader.test_closeness,
                                   period_feature=data_loader.test_period,
                                   trend_feature=data_loader.test_trend,
                                   target=data_loader.test_y,
                                   external_feature=data_loader.test_ef,
                                   sequence_length=data_loader.test_sequence_len)

# Compute metric
test_rmse = metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                        target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0)

# Evaluate
val_loss = ST_ResNet_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])
best_val_loss = data_loader.normalizer.min_max_denormal(best_val_loss)

print('Best val result', best_val_loss)
print('Test result', test_rmse)

print('Converged using %.2f hour' % ((val_loss[-1][0] - val_loss[0][0]) / 3600))
if nni_params:
    nni.report_final_result({
        'default': best_val_loss,
        'test-rmse': test_rmse
    })