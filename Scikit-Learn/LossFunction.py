__author__ = 'Aubrey'

import numpy as np
from sklearn import metrics

def compute_loss_function(y_pred, y_actual, target_ids, loss_function):
    return metrics.mean_squared_error(y_pred, y_actual)