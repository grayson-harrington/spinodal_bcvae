import numpy as np


def mae(y_true, y_pred, axis=None):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=axis)
