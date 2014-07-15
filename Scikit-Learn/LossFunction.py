__author__ = 'Aubrey'

import numpy as np
from sklearn import metrics
import math




class LossFunction:
    MEAN_SQUARED_ERROR = 0
    NDCG = 1
    PRECISION = 2

    def __init__(self, loss_function_type):
        self.loss_function_type = loss_function_type

    def __str__(self):
        return LossFunction.get_loss_function_short_name(self.loss_function_type)

    def get_display_name(self):
        return LossFunction.get_loss_function_display_name(self.loss_function_type)

    @staticmethod
    def get_loss_function_short_name(loss_function):
        name_dictionary = {
            LossFunction.MEAN_SQUARED_ERROR: 'MSE',
            LossFunction.NDCG: 'NDCG',
            LossFunction.PRECISION: 'P'
        }
        return name_dictionary[loss_function]

    @staticmethod
    # NDCG and Precision are "1 - ..." just so they're consistent with MSE in that "Lower is better"
    def get_loss_function_display_name(loss_function):
        name_dictionary = {
            LossFunction.MEAN_SQUARED_ERROR: 'Mean Squared Error',
            LossFunction.NDCG: '1 - Normalized Discounted Cumulative Gain@k',
            LossFunction.PRECISION: '1 - Precision@k'
        }
        return name_dictionary[loss_function]

    @staticmethod
    def compute_loss_function(y_pred,
                              y_actual,
                              target_ids,
                              loss_function,
                              k=5,
                              use_weights=True):
        if loss_function == LossFunction.MEAN_SQUARED_ERROR:
            val = metrics.mean_squared_error(y_pred, y_actual)
        else:
            target_ids_set = set(target_ids.tolist())
            scores = np.zeros(len(target_ids_set))
            for index, target_id in enumerate(target_ids_set):
                target_inds = target_ids == target_id
                y_pred_target = y_pred[target_inds]
                y_actual_target = y_actual[target_inds]
                pred_order = y_pred_target.argsort()[::-1]
                relevance = y_actual_target
                if not use_weights:
                    top_k_inds = y_actual_target.argsort()[:-k - 1:-1]
                    relevance = np.zeros(len(y_pred_target))
                    relevance[top_k_inds] = 1
                pred_ranking = relevance[pred_order]
                if loss_function == LossFunction.PRECISION:
                    scores[index] = LossFunction._precision_at_k(pred_ranking, k)
                else:
                    scores[index] = LossFunction._ndcg_at_k(pred_ranking, k)
            scores = 1 - scores
            val = scores.mean()
        return val

    @staticmethod
    def _get_top_k_vals(vals, k):
        sorted_inds = vals.argsort()
        top_k_inds = sorted_inds[:-k - 1:-1]
        top_k_vals = vals[top_k_inds]
        return top_k_vals

    @staticmethod
    def _dcg_at_k(vals, k=-1):
        if k < 1:
            k = len(vals)
        k_vals = vals[1:k]
        dcg = 0
        for index, val in enumerate(k_vals.tolist()):
            dcg += (2 ** val - 1) / math.log(index + 2, 2)
        return dcg

    @staticmethod
    def _idcg_at_k(vals, k):
        top_k_vals = LossFunction._get_top_k_vals(vals, k)
        return LossFunction._dcg_at_k(top_k_vals)

    @staticmethod
    def _ndcg_at_k(vals, k):
        dcg = LossFunction._dcg_at_k(vals, k)
        idcg = LossFunction._idcg_at_k(vals, k)
        return dcg / idcg

    @staticmethod
    def _ideal_score_at_k(vals, k):
        top_k_vals = LossFunction._get_top_k_vals(vals, k)
        return top_k_vals.sum()

    @staticmethod
    def _precision_at_k(vals, k=-1):
        if k < 1:
            k = len(vals)
        max_score = LossFunction._ideal_score_at_k(vals, k)
        top_k = vals[:k]
        score = top_k.sum()
        return score / max_score

class Scorer(object):
    @staticmethod
    def create_scorer(loss_function,target_ids):
        scorer = Scorer()
        scorer.loss_function = loss_function

    def __call__(self, *args):
        estimator = args[0]
        X = args[1]
        y = args[2]
        self._compute_score(estimator,X,y)

    def _compute_score(self,estimator,X,y):
        assert False
        y_pred = estimator.predict(X)
        if self.loss_function == LossFunction.MEAN_SQUARED_ERROR:
            pass

