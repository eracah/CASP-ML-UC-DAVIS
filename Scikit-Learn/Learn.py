__author__ = 'Evan and Aubrey'


import time
import numpy as np
import pickle
import os.path
import copy
import random
# from mpi4py import MPI
from sklearn import cross_validation
from sklearn.grid_search import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from Results import MainResult
from Results import FoldData
from LossFunction import LossFunction
from Estimator import Estimator, ScikitLearnEstimator
import HelperFunctions
import functools
# from pyspark import SparkContext

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

from Data import Data

def _train_and_test_with_params_args(args):
    return _train_and_test_with_params(*args)

def _train_and_test_with_params(configs,
                                train_x, train_y, train_target_ids,
                                test_x, test_y, test_target_ids,
                                estimator_configs,
                                estimator_params):
        estimator = copy.deepcopy(estimator_configs.estimator)
        estimator.file_id = random.randint(0,2**64)
        estimator.set_params(**estimator_params)
        #TODO: Do we want this?
        # estimator.set_cv_data(test_x, test_y, test_target_ids)
        estimator.fit(train_x, train_y, train_target_ids)
        cv_test_pred = estimator.predict(test_x, test_y, test_target_ids)
        score = LossFunction.compute_loss_function(cv_test_pred, test_y,
                                                   test_target_ids, configs.cv_loss_function)
        return score

class Learn():
    def __init__(self, configs, spark_context):
        self.configs = configs
        self.spark_context = spark_context
        self.data = self._get_data(self.configs)
        self.estimator_configs = configs.estimator_configs
        self.estimator_name = configs.estimator_name
        # instantiate a main result object
        self.main_results = MainResult(configs)
        self.main_results.data = self.data

        if self.configs.normalize_data:
            self.normalizer = StandardScaler()
        else:
            self.normalizer = StandardScaler(with_mean=False, with_std=False)

        if isinstance(self.estimator_configs.estimator, ScikitLearnEstimator) and \
                isinstance(self.estimator_configs.estimator.skl_estimator, RandomForestRegressor):
            cores = {'n_jobs': self.configs.n_cores}
            self.estimator_configs.estimator.set_params(**cores)


    def run_grid_search(self):
        for training_size in self.configs.training_sizes:
            print 'training size: ', training_size
            for trial in range(self.configs.trials_per_size):
                x_train, y_train, train_indices, train_targets = self.data.sample_train(training_size)
                best_estimator, best_params, cv_time = self._get_best_estimator_and_params(train_targets)
                best_estimator, train_time = self._train_data(best_estimator, x_train, y_train, train_indices)
                best_estimator.predict(x_train, y_train, self.data.get_target_ids(train_indices))
                self._print_data(trial, best_params, train_time, cv_time)
                fold_data = FoldData(training_size, best_estimator, train_indices, train_targets,
                                     train_time, cv_time, self.normalizer)
                self.main_results.add_estimator_results(fold_data,
                                                        trial,
                                                        self.configs.trials_per_size)

        if self.configs.save_the_results:
            self.main_results.save_data()

        return self.main_results


    def _print_data(self, trial, best_params, train_time, cv_time):
        print 'trial: ', trial+1, ' of ', self.configs.trials_per_size
        print '\tbest parameter values: ', best_params
        print '\ttrain_time: ', train_time
        print '\tcv time: ', cv_time

    def _train_data(self, best_estimator, x_train, y_train, train_indices):
        train_time_start = time.time()
        best_estimator.clear_cv_data()
        x_train = self.normalizer.fit_transform(x_train)
        best_estimator.fit(x_train, y_train, self.data.get_target_ids(train_indices))
        train_time = time.time() - train_time_start
        return best_estimator, train_time

    def _get_best_estimator_and_params(self,train_targets):
        cv_time_start = time.time()
        param_grid = list(ParameterGrid(self.configs.estimator_configs.params))
        sampled_targets = np.asarray(list(set(train_targets)), dtype=int)
        kf = list(cross_validation.KFold(len(sampled_targets), self.configs.n_folds, shuffle=True))
        cv_scores = []
        for i in range(len(kf)):
            cv_scores.append([0] * len(param_grid))
        if len(param_grid) == 1 and False:
            best_param_index = 0
        else:
            best_param_index = self._cross_validate(param_grid, kf, sampled_targets, cv_scores)

        best_params = param_grid[best_param_index]
        best_estimator = copy.deepcopy(self.configs.estimator_configs.estimator)
        best_estimator.set_params(**best_params)
        cv_time = time.time() - cv_time_start
        return best_estimator, best_params, cv_time

    def _cross_validate(self, param_grid, kf, sampled_targets, cv_scores):
            for param_index in range(len(param_grid)):
                params = param_grid[param_index]
                func_params = []
                for kf_index in range(len(kf)):
                    train_target_indices, test_target_indices = kf[kf_index]
                    cv_train_targets = sampled_targets[train_target_indices]
                    cv_test_targets = sampled_targets[test_target_indices]

                    cv_train_x, cv_train_y, _, cv_train_target_ids = self.data.select_targets(cv_train_targets)
                    cv_test_x, cv_test_y, _, cv_test_target_ids = self.data.select_targets(cv_test_targets)

                    cv_train_x = self.normalizer.fit_transform(cv_train_x)
                    cv_test_x = self.normalizer.transform(cv_test_x)

                    p = [self.configs,
                         cv_train_x, cv_train_y, cv_train_target_ids,
                         cv_test_x, cv_test_y, cv_test_target_ids,
                         self.configs.estimator_configs, params]
                    func_params.append(p)
                kf_scores = []
                if self.spark_context == None:
                    for params in func_params:
                        score = _train_and_test_with_params_args(params)
                        kf_scores.append(score)
                else:
                    dist_func_params = self.spark_context.parallelize(func_params)
                    dist_scores = dist_func_params.map(_train_and_test_with_params_args)
                    all_scores = dist_scores.collect()
                    for score in all_scores:
                        kf_scores.append(score)

                for kf_index, kf_score in enumerate(kf_scores):
                    cv_scores[kf_index][param_index] = kf_score
            best_param_index = self._get_best_param_index(cv_scores)
            return best_param_index

    def _get_best_param_index(self, all_scores):
        mean_scores = []
        for index in range(len(all_scores[0])):
            param_scores = [fold_scores[index] for fold_scores in all_scores ]
            mean_scores.append(np.asarray(param_scores).mean())
        return np.asarray(mean_scores).argmin()

    @staticmethod
    def _get_data(configs):
        data_file_name = configs.target_data_file_name
        load_target_data = configs.load_target_data
        file_exists = os.path.isfile(data_file_name)
        if load_target_data and file_exists:
            with open(data_file_name, 'rb') as f:
                data = pickle.load(f)
        else:
            data = Data(configs)
            HelperFunctions.save_object(data, data_file_name)
        return data



