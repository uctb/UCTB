import xgboost as xgb


class XGBoost(object):

    def __init__(self, max_depth=5, verbosity=0, objective='reg:linear', eval_metric='rmse'):
        self.param = {
            'max_depth': max_depth,
            'verbosity ': verbosity,
            'objective': objective,
            'eval_metric': eval_metric
        }

    def fit(self, X, y, num_boost_round=5):

        train_data = xgb.DMatrix(X, label=y)

        watchlist = [(train_data, 'train')]

        self.bst = xgb.train(self.param, train_data, num_boost_round, watchlist)

    def predict(self, X):

        test_data = xgb.DMatrix(X)

        return self.bst.predict(test_data)