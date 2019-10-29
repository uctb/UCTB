import numpy as np
from hmmlearn import hmm


class HMM(object):
    def __init__(self, num_components, n_iter, hmm_kernal=hmm.GaussianHMM):

        self._num_components = num_components
        self._iter = n_iter

        self._hmm = hmm_kernal(n_components=self._num_components, n_iter=self._iter, covariance_type='full')

    def fit(self, x):
        self._hmm.fit(x)
        if self._hmm.monitor_.converged:
            print('Status: converged')

    def predict(self, x, length):
        # predict the state for each element of X
        # and store the last state
        last_state = self._hmm.predict_proba(x)[-1:]

        pre_state = []
        pre_observation = []

        for i in range(length):
            # predict the state of next moment using the transmat
            last_state = np.dot(last_state, self._hmm.transmat_)

            pre_state.append(last_state)

            # dot product between the state-probability and state-means
            pre_observation.append([np.dot(last_state, self._hmm.means_)[0][0]])

        return pre_observation