import nni
import GPUtil

from UCTB.dataset import GridTrafficLoader
from UCTB.model import ST_ResNet
from UCTB.evaluation import metric

args = {
    'dataset': 'DiDi',
    'city': 'Xian',
    'num_residual_unit': 4,
    'conv_filters': 64,
    'kernel_size': 3,
    'lr': 1e-5,
    'batch_size': 32,
    'MergeIndex': 6
}

code_version = 'ST_ResNet_{}_{}_F{}'.format(args['dataset'], args['city'], int(args['MergeIndex'])*5)

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    code_version += ('_' + str(nni_sid))

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    if nni_params:
        current_device = str(deviceIDs[int(nni_sid) % len(deviceIDs)])
    else:
        current_device = str(deviceIDs[0])

# Config data loader
data_loader = GridTrafficLoader(dataset=args['dataset'], city=args['city'], closeness_len=6, period_len=7, trend_len=4, MergeIndex=args['MergeIndex'])

ST_ResNet_Obj = ST_ResNet(closeness_len=data_loader.closeness_len,
                          period_len=data_loader.period_len,
                          trend_len=data_loader.trend_len,
                          external_dim=data_loader.external_dim, lr=args['lr'],
                          num_residual_unit=args['num_residual_unit'], conv_filters=args['conv_filters'],
                          kernel_size=args['kernel_size'], width=data_loader.width, height=data_loader.height,
                          gpu_device=current_device, code_version=code_version)

ST_ResNet_Obj.build()

print(args['dataset'], args['city'], code_version)
print('Number of trainable variables', ST_ResNet_Obj.trainable_vars)
print('Number of training samples', data_loader.train_sequence_len)

print('debug')

# Training
ST_ResNet_Obj.fit(closeness_feature=data_loader.train_closeness,
                  period_feature=data_loader.train_period,
                  trend_feature=data_loader.train_trend,
                  target=data_loader.train_y,
                  external_feature=data_loader.train_ef,
                  sequence_length=data_loader.train_sequence_len,
                  batch_size=args['batch_size'], early_stop_length=200,
                  validate_ratio=0.1)

# Predict
prediction = ST_ResNet_Obj.predict(closeness_feature=data_loader.test_closeness,
                                   period_feature=data_loader.test_period,
                                   trend_feature=data_loader.test_trend,
                                   target=data_loader.test_y,
                                   external_feature=data_loader.test_ef,
                                   sequence_length=data_loader.test_sequence_len)

# Compute metric
test_rmse = metric.rmse(prediction=data_loader.normalizer.inverse_transform(prediction['prediction']),
                        target=data_loader.normalizer.inverse_transform(data_loader.test_y))

# Evaluate
val_loss = ST_ResNet_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])
best_val_loss = data_loader.normalizer.inverse_transform(best_val_loss)

print('Best val result', best_val_loss)
print('Test result', test_rmse)

print('Converged using %.2f hour' % ((val_loss[-1][0] - val_loss[0][0]) / 3600))
if nni_params:
    nni.report_final_result({
        'default': best_val_loss,
        'test-rmse': test_rmse
    })