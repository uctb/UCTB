import numpy as np


def one_hot(x):
    if isinstance(x, np.ndarray) is False:
        x = np.array(x)
    x = (np.arange(np.max(x)+1) == x[:, None]).astype(np.integer)
    return np.concatenate(x, axis=0)
