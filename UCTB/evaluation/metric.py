import numpy as np


def rmse(prediction, target, threshold=None):
    """
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target
            will be removed in computing the rmse
    """
    if threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]


def mape(prediction, target, threshold=0):
    """
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the mape,
            the smallest threshold in mape is zero.
    """
    assert threshold > 0
    return (np.dot((np.abs(prediction - target) / (target + (1 - (target > threshold)))).reshape([1, -1]),
                   target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]


# def rmse_grid(prediction, target, threshold=None):
#     if threshold is None:
#         return np.sqrt(np.mean(np.square(prediction - target), axis=0))
#     else:
#         return np.sqrt(np.sum(np.square(prediction - target) *
#                               (target > threshold), axis=0) / np.sum(target > threshold, axis=0))
#
#
# def mape_grid(prediction, target, threshold=0):
#     assert threshold > 0
#     return np.sum((np.abs(prediction - target) / (target + (1 - (target > threshold))))
#                   * (target > threshold), axis=0) / np.sum(target > threshold, axis=0)