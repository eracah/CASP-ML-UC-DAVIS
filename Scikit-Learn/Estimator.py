__author__ = 'Aubrey'

import abc
import numpy as np
import random
from RankLib.RankLib import save_in_letor_format
from RankLib.RankLib import load_letor_scores
from RankLib.RankLib import run_ranking
from RankLib.RankLib import RankLibConfigs
from HelperFunctions import check_input

import Learn

class Estimator(object):
    __metaclass__ = abc.ABCMeta

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def fit(self, X, Y, target_ids):
        return

    @abc.abstractmethod
    def predict(self, X, Y, target_ids):
        return

    @abc.abstractmethod
    def get_name(self):
        return

    def set_cv_data(self, X, Y, target_ids):
        return

    def clear_cv_data(self):
        return

class GuessEstimator(Estimator):
    def fit(self, X, Y, target_ids):
        return

    def predict(self, X, Y, target_ids):
        Estimator.check_input(X, Y, target_ids)
        scores = np.ndarray(len(Y))
        for i in range(len(scores)):
            scores[i] = random.random()
        return scores

    def get_name(self):
        return 'Guess'


train_file_name = 'RankLib/Data/letor_train.txt'
test_file_name = 'RankLib/Data/letor_test.txt'
model_file_name = 'RankLib/Data/letor_model.txt'
score_file_name = 'RankLib/Data/letor_score.txt'
cv_file_name = 'RankLib/Data/letor_cv.txt'

class RankLib(Estimator):
    def __init__(self):
        pass

    def fit(self, X, Y, target_ids):
        k = self.k
        check_input(X, Y, target_ids)
        Y_01 = Learn.Data.make_score_0_1(Y, target_ids, k)
        save_in_letor_format(X, Y_01, target_ids, train_file_name)
        rl_configs = RankLibConfigs()
        rl_configs.train_file_name = train_file_name
        rl_configs.save_file_name = model_file_name
        rl_configs.k = self.k
        if hasattr(self, 'cv_file_name'):
            rl_configs.cv_file_name = self.cv_file_name
        run_ranking(rl_configs)

    def set_cv_data(self, X, Y, target_ids):
        k = self.k
        self.cv_file_name = cv_file_name
        check_input(X, Y, target_ids)
        Y_01 = Learn.Data.make_score_0_1(Y, target_ids, k)
        save_in_letor_format(X, Y_01, target_ids, cv_file_name)


    def clear_cv_data(self):
        del self.cv_file_name

    def predict(self, X, Y, target_ids):
        k = self.k
        check_input(X, Y, target_ids)
        Y_01 = Learn.Data.make_score_0_1(Y, target_ids, k)
        save_in_letor_format(X, Y_01, target_ids, test_file_name)
        rl_configs = RankLibConfigs()
        rl_configs.model_file_name = model_file_name
        rl_configs.test_file_name = test_file_name
        rl_configs.score_file_name = score_file_name
        rl_configs.k = self.k
        run_ranking(rl_configs)
        return load_letor_scores(rl_configs.score_file_name, len(Y))

    def get_name(self):
        return 'RankLib'