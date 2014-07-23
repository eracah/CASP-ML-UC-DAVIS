__author__ = 'Aubrey'

import abc
import numpy as np
import random
import copy
import multiprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
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

    def get_short_name(self):
        return self.get_name()

    def set_cv_data(self, X, Y, target_ids):
        return

    def clear_cv_data(self):
        return


class ScikitLearnEstimator(Estimator):
    _short_name_dict = {
        'KNeighborsRegressor': 'KNR',
        'RandomForestRegressor': 'RFR',
        'AdaBoostRegressor': 'ABR',
        'ExtraTreesRegressor': 'ETR',
        'GradientBoostingRegressor': 'GBR',
        'RadiusNeighborsRegressor': 'RNR'
    }

    def __init__(self, skl_estimator):
        self.skl_estimator = skl_estimator

    def fit(self, X, Y, target_ids):
        self.skl_estimator.fit(X, Y)

    def predict(self, X, Y, target_ids):
        check_input(X, Y, target_ids)
        return self.skl_estimator.predict(X)

    def get_name(self):
        return 'SciKitLearn-' + self._get_skl_estimator_name()

    def get_short_name(self):
        return 'SKL-' + ScikitLearnEstimator._short_name_dict[self._get_skl_estimator_name()]

    def _get_skl_estimator_name(self):
        return repr(self.skl_estimator).split('(')[0]

    def set_params(self, **kwargs):
        self.skl_estimator.set_params(**kwargs)

class ScitkitLearnKNeighborsRegressor(ScikitLearnEstimator):
    def __init__(self):
        self.skl_estimator = KNeighborsRegressor()

class ScitkitLearnRandomForestRegressor(ScikitLearnEstimator):

    def __init__(self, n_estimators=16):
        self.skl_estimator = RandomForestRegressor(n_estimators=n_estimators)

class GuessEstimator(Estimator):
    def fit(self, X, Y, target_ids):
        return

    def predict(self, X, Y, target_ids):
        check_input(X, Y, target_ids)
        scores = np.ndarray(len(Y))
        for i in range(len(scores)):
            scores[i] = random.random()
        return scores

    def get_name(self):
        return 'Guess'


train_file_name = 'RankLib/Data/letor_train_'
test_file_name = 'RankLib/Data/letor_test_'
model_file_name = 'RankLib/Data/letor_model_'
score_file_name = 'RankLib/Data/letor_score_'
cv_file_name = 'RankLib/Data/letor_cv_'

def get_unique_file_name(file_name):
    current = multiprocessing.current_process()
    return file_name + current.name + '.txt'

class RankLib(Estimator):
    _ranker_opt_name_dict = {
        0: 'MART',
        1: 'RankNet',
        2: 'RankBoost',
        3: 'AdaRank',
        4: 'Coordinate Ascent',
        6: 'LambdaMART',
        7: 'ListNet',
        8: 'Random Forests'
    }

    _ranker_opt_short_name_dict = {
        0: 'MART',
        1: 'RN',
        2: 'RB',
        3: 'AR',
        4: 'CA',
        6: 'LM',
        7: 'LN',
        8: 'RF'
    }

    def __init__(self,
                 ranker_opt=0,
                 make_score_binary_at_k=False,
                 k=5,
                 tree=200,
                 shrinkage=.1,
                 estop=50):
        self.rl_configs = RankLibConfigs()
        self.rl_configs.k = k
        self.rl_configs.ranker_opt = ranker_opt
        self.rl_configs.make_score_binary_at_k = make_score_binary_at_k
        self.rl_configs.tree = tree
        self.rl_configs.shrinkage = shrinkage
        self.rl_configs.estop = estop

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.rl_configs, k, v)

    def fit(self, X, Y, target_ids):
        current = multiprocessing.current_process()
        check_input(X, Y, target_ids)
        if self.rl_configs.make_score_binary_at_k:
            Y = Learn.Data.make_score_0_1(Y, target_ids, self.rl_configs.k)
        train_file = get_unique_file_name(train_file_name)
        model_file = get_unique_file_name(model_file_name)
        save_in_letor_format(X, Y, target_ids, train_file)
        rl_configs = copy.deepcopy(self.rl_configs)
        rl_configs.train_file_name = train_file
        rl_configs.save_file_name = model_file
        if hasattr(self, 'cv_file_name'):
            rl_configs.cv_file_name = self.cv_file_name
        run_ranking(rl_configs)

    def set_cv_data(self, X, Y, target_ids):
        self.cv_file_name = get_unique_file_name(cv_file_name)
        check_input(X, Y, target_ids)
        if self.rl_configs.make_score_binary_at_k:
            Y = Learn.Data.make_score_0_1(Y, target_ids, self.rl_configs.k)
        save_in_letor_format(X, Y, target_ids, self.cv_file_name)


    def clear_cv_data(self):
        if hasattr(self, 'cv_file_name'):
            del self.cv_file_name

    def predict(self, X, Y, target_ids):
        model_file = get_unique_file_name(model_file_name)
        test_file = get_unique_file_name(test_file_name)
        score_file = get_unique_file_name(score_file_name)
        check_input(X, Y, target_ids)
        if self.rl_configs.make_score_binary_at_k:
            Y = Learn.Data.make_score_0_1(Y, target_ids, self.rl_configs.k)
        save_in_letor_format(X, Y, target_ids, test_file)
        rl_configs = copy.deepcopy(self.rl_configs)
        rl_configs.model_file_name = model_file
        rl_configs.test_file_name = test_file
        rl_configs.score_file_name = score_file
        run_ranking(rl_configs)
        return load_letor_scores(rl_configs.score_file_name, len(Y))

    def get_name(self):
        return 'RankLib-' + RankLib._ranker_opt_name_dict[self.ranker_opt]

    def get_short_name(self):
        return 'RL-' + RankLib._ranker_opt_short_name_dict[self.ranker_opt]

    @property
    def ranker_opt(self):
        return self.rl_configs.ranker_opt

    @ranker_opt.setter
    def ranker_opt(self, value):
        self.rl_configs.ranker_opt = value