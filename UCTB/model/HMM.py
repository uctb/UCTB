import numpy as np
from hmmlearn import hmm

from .ModelObject import ModelObject


class HMM(ModelObject):

    def __init__(self, num_components=16, n_iter=1000, hmm_kernel=hmm.GaussianHMM, predict_length=1):
        super(HMM, self).__init__()
        self.num_components = num_components
        self.n_iter = n_iter
        self.hmm_kernel = hmm_kernel
        self.predict_length = predict_length

    def fit(self, X, y=None):

        self.models = []
        train_x, train_y = self.make_train_data(X, y)
        closeness_feature, period_feature, trend_feature = train_x
        node_num = train_y.shape[1]
        for station_index in range(node_num):
            self.models.append([])
            for _ in range(3):
                self.models[-1].append(self.hmm_kernel(n_components=self.num_components, n_iter=self.n_iter,
                                                       covariance_type='full'))
            if closeness_feature is not None and 0 not in closeness_feature.shape:
                self.models[-1][0].fit(closeness_feature[:, station_index: station_index + 1, -1, 0])
            if period_feature is not None and 0 not in period_feature.shape:
                self.models[-1][1].fit(period_feature[:, station_index: station_index + 1, -1, 0])
            if trend_feature is not None and 0 not in trend_feature.shape:
                self.models[-1][2].fit(trend_feature[:, station_index: station_index + 1, -1, 0])
        return self.models

    @staticmethod
    def _predict(model, X, length):

        # predict the state for each element of X
        # and store the last state
        last_state = model.predict_proba(X)[-1:]

        pre_state = []
        pre_observation = []

        for i in range(length):
            # predict the state of next moment using the transmat
            last_state = np.dot(last_state, model.transmat_)

            pre_state.append(last_state)

            # dot product between the state-probability and state-means
            pre_observation.append([np.dot(last_state, model.means_)[0][0]])

        return pre_observation

    def predict(self, X):

        closeness_feature, period_feature, trend_feature = self.make_test_data(X)
        slot_num = 0
        node_num = 0
        feature_num = 0
        if closeness_feature is not None and 0 not in closeness_feature.shape:
            slot_num = closeness_feature.shape[0]
            node_num = closeness_feature.shape[1]
            feature_num += 1
        if period_feature is not None and 0 not in period_feature.shape:
            slot_num = period_feature.shape[0]
            node_num = period_feature.shape[1]
            feature_num += 1
        if trend_feature is not None and 0 not in trend_feature.shape:
            slot_num = trend_feature.shape[0]
            node_num = trend_feature.shape[1]
            feature_num += 1
        self.results = np.zeros((node_num, slot_num), dtype=np.float32)

        for i in range(node_num):
            for j in range(slot_num):
                if closeness_feature is not None and 0 not in closeness_feature.shape:
                    self.results[i][j] += self._predict(self.models[i][0], closeness_feature[j, i, :, :],
                                                        self.predict_length)
                if period_feature is not None and 0 not in period_feature.shape:
                    self.results[i][j] += self._predict(self.models[i][1], period_feature[j, i, :, :],
                                                        self.predict_length)
                if trend_feature is not None and 0 not in trend_feature.shape:
                    self.results[i][j] += self._predict(self.models[i][2], trend_feature[j, i, :, :],
                                                        self.predict_length)
        self.results /= feature_num
        return np.expand_dims(np.transpose(self.results, (1, 0)), 2)

