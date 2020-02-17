import nni

from UCTB.dataset import GridTrafficLoader
from UCTB.model import DeepST
from UCTB.evaluation import metric

args = {
    'dataset': 'DiDi',
    'city': 'Xian',
    'num_conv_filters': 64,
    'kernel_size': 3,
    'lr': 5e-5,
    'batch_size': 64,
    'MergeIndex': 6,
}

code_version = 'DeepST_{}_{}_F{}'.format(args['dataset'], args['city'], int(args['MergeIndex'])*5)

nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    code_version += ('_' + str(nni_sid))

# Config data loader
data_loader = GridTrafficLoader(dataset=args['dataset'], city=args['city'],
                                closeness_len=6, period_len=7, trend_len=4,MergeIndex=args['MergeIndex'])

deep_st_obj = DeepST(closeness_len=data_loader.closeness_len,
                     period_len=data_loader.period_len,
                     trend_len=data_loader.trend_len,
                     external_dim=data_loader.external_dim,
                     num_conv_filters=args['num_conv_filters'], kernel_size=args['kernel_size'],
                     code_version=code_version,
                     width=data_loader.width, height=data_loader.height, lr=args['lr'])

deep_st_obj.build()

print('Trainable variables', deep_st_obj.trainable_vars)

# Training
deep_st_obj.fit(closeness_feature=data_loader.train_closeness,
                period_feature=data_loader.train_period,
                trend_feature=data_loader.train_trend,
                target=data_loader.train_y,
                external_feature=data_loader.train_ef,
                sequence_length=data_loader.train_sequence_len,
                batch_size=args['batch_size'],
                validate_ratio=0.1)

# Predict
prediction = deep_st_obj.predict(closeness_feature=data_loader.test_closeness,
                                 period_feature=data_loader.test_period,
                                 trend_feature=data_loader.test_trend,
                                 target=data_loader.test_y,
                                 external_feature=data_loader.test_ef,
                                 sequence_length=data_loader.test_sequence_len)

test_rmse = metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                        target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0)

# Compute metric
print('Test result', test_rmse)

# Evaluate
val_loss = deep_st_obj.load_event_scalar('val_loss')

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