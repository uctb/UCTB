import numpy as np
from hmmlearn import hmm


class HMM(object):
    def __init__(self, num_components, n_iter, hmm_kernal=hmm.GaussianHMM):

        self.__num_components = num_components
        self.__iter = n_iter

        self.__hmm = hmm_kernal(n_components=self.__num_components,
                                n_iter=self.__iter,
                                covariance_type='full')

    def fit(self, X):
        self.__hmm.fit(X)
        if self.__hmm.monitor_.converged:
            print('Status: converged')
    
    def predict(self, X, length):
        # predict the state for each element of X
        # and store the last state
        last_state = self.__hmm.predict_proba(X)[-1:]

        pre_state = []
        pre_observation = []

        for i in range(length):
            # predict the state of next moment using the transmat
            last_state = np.dot(last_state, self.__hmm.transmat_)

            pre_state.append(last_state)

            # dot product between the state-probability and state-means
            pre_observation.append(np.dot(last_state, self.__hmm.means_)[0][0])

        return pre_observation