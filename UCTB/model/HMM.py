import numpy as np
from hmmlearn import hmm


class HMM(object):
    def __init__(self, closeness_feature, period_feature, trend_feature,
                 num_components, n_iter, hmm_kernal=hmm.GaussianHMM):

        self.temporal_features = []
        if closeness_feature is not None and 0 not in closeness_feature.shape:
            self.temporal_features.append(closeness_feature)
        if period_feature is not None and 0 not in period_feature.shape:
            self.temporal_features.append(period_feature)
        if trend_feature is not None and 0 not in trend_feature.shape:
            self.temporal_features.append(trend_feature)

        self._num_components = num_components
        self._iter = n_iter

        self._hmm = hmm_kernal(n_components=self._num_components, n_iter=self._iter, covariance_type='full')

    def fit(self, X):
        self._hmm.fit(X)
        if self._hmm.monitor_.converged:
            print('Status: converged')
    
    def predict(self, X, length):
        # predict the state for each element of X
        # and store the last state
        last_state = self._hmm.predict_proba(X)[-1:]

        pre_state = []
        pre_observation = []

        for i in range(length):
            # predict the state of next moment using the transmat
            last_state = np.dot(last_state, self._hmm.transmat_)

            pre_state.append(last_state)

            # dot product between the state-probability and state-means
            pre_observation.append([np.dot(last_state, self._hmm.means_)[0][0]])

        return pre_observation