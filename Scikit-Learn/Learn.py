__author__ = 'Evan and Aubrey'


import time
import numpy as np
import pickle
import os.path
import copy
from sklearn import cross_validation
from sklearn.grid_search import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from Results import MainResult
from Results import FoldData
from LossFunction import LossFunction
from Estimator import Estimator, ScikitLearnEstimator
import HelperFunctions
import functools

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

from Data import Data

def _train_and_test_with_params_args(args):
    return _train_and_test_with_params(*args)

def _train_and_test_with_params(learn,
                                train_x, train_y, train_target_ids,
                                test_x, test_y, test_target_ids,
                                estimator_configs,
                                estimator_params):
        estimator = copy.deepcopy(estimator_configs.estimator)
        estimator.set_params(**estimator_params)
        #TODO: Do we want this?
        # estimator.set_cv_data(test_x, test_y, test_target_ids)
        estimator.fit(train_x, train_y, train_target_ids)
        cv_test_pred = estimator.predict(test_x, test_y, test_target_ids)
        score = LossFunction.compute_loss_function(cv_test_pred, test_y,
                                                   test_target_ids, learn.configs.cv_loss_function)
        return score

class Learn():
    def __init__(self, configs):
        self.configs = configs

        self.data = self._get_data(self.configs)
        self.estimator_configs = configs.estimator_configs
        self.estimator_name = configs.estimator_name
        # instantiate a main result object
        self.main_results = MainResult(configs)
        self.main_results.data = self.data

        if isinstance(self.estimator_configs.estimator, ScikitLearnEstimator) and \
                isinstance(self.estimator_configs.estimator.skl_estimator, RandomForestRegressor):
            cores = {'n_jobs': self.configs.n_cores}
            self.estimator_configs.estimator.set_params(**cores)


    def run_grid_search(self):
        num_processes = self.configs.num_cv_processes
        use_cv_pool = self.configs.use_cv_pool
        # pool = Pool(processes=num_processes)
        for training_size in self.configs.training_sizes:
            print 'training size: ', training_size
            for trial in range(self.configs.trials_per_size):
                x_train, y_train, train_indices, train_targets = self.data.sample_train(training_size)
                num_folds = self.configs.n_folds
                cv_time_start = time.time()
                estimator_configs = self.configs.estimator_configs
                sampled_targets = np.asarray(list(set(train_targets)), dtype=int)
                kf = list(cross_validation.KFold(len(sampled_targets), num_folds, shuffle=True))

                params = estimator_configs.params
                param_grid = list(ParameterGrid(params))
                cv_scores = []
                for i in range(len(kf)):
                    cv_scores.append([0] * len(param_grid))

                normalizer = StandardScaler()
                if not self.configs.normalize_data:
                    normalizer = StandardScaler(with_mean=False, with_std=False)
                if len(param_grid) == 1 and False:
                    best_param_index = 0
                else:
                    # for train_target_indices, test_target_indices in kf:
                    for kf_index in range(len(kf)):
                        train_target_indices, test_target_indices = kf[kf_index]
                        cv_train_targets = sampled_targets[train_target_indices]
                        cv_test_targets = sampled_targets[test_target_indices]

                        cv_train_x, cv_train_y, _, cv_train_target_ids = self.data.select_targets(cv_train_targets)
                        cv_test_x, cv_test_y, _, cv_test_target_ids = self.data.select_targets(cv_test_targets)

                        cv_train_x = normalizer.fit_transform(cv_train_x)
                        cv_test_x = normalizer.transform(cv_test_x)

                        #TODO: estimator_configs.estimator.set_params('n_cores': configs.n_cores )

                        func_params = []
                        for param_index in range(len(param_grid)):
                            params = param_grid[param_index]
                            p = [self,
                                 cv_train_x, cv_train_y, cv_train_target_ids,
                                 cv_test_x, cv_test_y, cv_test_target_ids,
                                 estimator_configs, params]
                            func_params.append(p)

                        if use_cv_pool:
                            param_scores = pool.map(_train_and_test_with_params_args, func_params)
                        else:
                            param_scores = []
                            for param_index in range(len(param_grid)):
                                params = func_params[param_index]
                                score = _train_and_test_with_params_args(params)
                                param_scores.append(score)
                        cv_scores[kf_index] = param_scores
                    best_param_index = self._get_best_param_index(cv_scores)
                best_params = param_grid[best_param_index]
                best_estimator = copy.deepcopy(estimator_configs.estimator)
                best_estimator.set_params(**best_params)

                cv_time = time.time() - cv_time_start

                train_time_start = time.time()
                #fit estimator with best parameters to training data
                best_estimator.clear_cv_data()

                x_train = normalizer.fit_transform(x_train)
                best_estimator.fit(x_train, y_train, self.data.get_target_ids(train_indices))

                if hasattr(best_estimator, 'feature_importances_'):
                    feature_importances = best_estimator.feature_importances_
                else:
                    feature_importances = []
                train_time = time.time() - train_time_start

                best_estimator.predict(x_train, y_train, self.data.get_target_ids(train_indices))

                print 'trial: ', trial+1, ' of ', self.configs.trials_per_size
                print '\tbest parameter values: ', best_params
                print '\ttrain_time: ', train_time
                print '\tcv time: ', cv_time

                fold_data = FoldData()
                fold_data.training_size = training_size
                fold_data.estimator = best_estimator

                fold_data.train_inds = train_indices
                fold_data.train_targets = train_targets
                fold_data.train_time = train_time
                fold_data.cv_time = cv_time

                fold_data.normalizer = normalizer
                fold_data.feature_importances = feature_importances

                #add grid_search results to main results to be further processed
                self.main_results.add_estimator_results(fold_data,
                                                        trial,
                                                        self.configs.trials_per_size)

        if self.configs.save_the_results:
            self.main_results.save_data()

        return self.main_results

    def _train_and_test_with_params(self,
                                    train_x, train_y, train_target_ids,
                                    test_x, test_y, test_target_ids,
                                    estimator_configs,
                                    estimator_params):
        estimator = copy.deepcopy(estimator_configs.estimator)
        estimator.set_params(**estimator_params)
        estimator.set_cv_data(test_x, test_y, test_target_ids)
        estimator.fit(train_x, train_y, train_target_ids)
        cv_test_pred = estimator.predict(test_x, test_y, test_target_ids)
        score = LossFunction.compute_loss_function(cv_test_pred, test_y,
                                                   test_target_ids, self.configs.cv_loss_function)
        return score

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



