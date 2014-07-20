__author__ = 'Evan and Aubrey'


import time
import numpy as np
import pickle
import os.path
import copy
from sklearn import cross_validation
from sklearn.grid_search import ParameterGrid
from Results import MainResult
from Results import FoldData
from LossFunction import LossFunction
from Estimator import Estimator
import HelperFunctions

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool
import scipy.stats as ss

from Data import Data

class Learn():
    def __init__(self, config):
        self.config = config

        self.data = self._get_data(self.config)
        self.estimator_configs = config.estimator_configs
        self.estimator_name = config.estimator_name
        # instantiate a main result object
        self.main_results = MainResult(self.estimator_name, self.config.path_to_store_results,
                                       self.config.save_results_file_name)


    def run_grid_search(self):
        self.main_results.data = self.data
        for training_size in self.config.training_sizes:
            for trial in range(self.config.trials_per_size):
                x_train, y_train, train_indices, train_targets = self.data.sample_train(training_size)

                estimator_configs = self.estimator_configs

                num_folds = self.config.n_folds
                t0 = time.time()

                sampled_targets = np.asarray(list(set(train_targets)), dtype=int)
                kf = cross_validation.KFold(len(sampled_targets), num_folds, shuffle=True)

                params = estimator_configs.params
                param_grid = list(ParameterGrid(params))
                cv_scores = []
                for i in range(len(param_grid)):
                    cv_scores.append([])

                normalizer = StandardScaler()
                if not self.config.normalize_data:
                    normalizer = StandardScaler(with_mean=False, with_std=False)
                if len(param_grid) == 1:
                    best_param_index = 0
                else:
                    for train_target_indices, test_target_indices in kf:
                        cv_train_targets = sampled_targets[train_target_indices]
                        cv_test_targets = sampled_targets[test_target_indices]

                        cv_train_x, cv_train_y, _, cv_train_target_ids = self.data.select_targets(cv_train_targets)
                        cv_test_x, cv_test_y, _, cv_test_target_ids = self.data.select_targets(cv_test_targets)

                        cv_train_x = normalizer.fit_transform(cv_train_x)
                        cv_test_x = normalizer.transform(cv_test_x)

                        #TODO: estimator_configs.estimator.set_params('n_cores': configs.n_cores )
                        for param_index, params in enumerate(param_grid):
                            estimator_configs.estimator.set_params(**params)
                            if isinstance(estimator_configs.estimator, Estimator):
                                estimator_configs.estimator.set_cv_data(cv_test_x,
                                                                        cv_test_y,
                                                                        cv_test_target_ids)
                                estimator_configs.estimator.fit(cv_train_x,
                                                                cv_train_y,
                                                                cv_train_target_ids)
                                cv_test_pred = estimator_configs.estimator.predict(cv_test_x,
                                                                                   cv_test_y,
                                                                                   cv_test_target_ids)
                            else:
                                estimator_configs.estimator.fit(cv_train_x, cv_train_y)
                                cv_test_pred = estimator_configs.estimator.predict(cv_test_x)
                            score = LossFunction.compute_loss_function(cv_test_pred, cv_test_y,
                                                                       cv_test_target_ids, self.config.cv_loss_function)
                            cv_scores[param_index].append(score)
                        best_param_index = self._get_best_param_index(cv_scores)
                best_params = param_grid[best_param_index]
                best_estimator = copy.deepcopy(estimator_configs.estimator)
                best_estimator.set_params(**best_params)

                search_time = time.time() - t0

                t0 = time.time()
                #fit estimator with best parameters to training data
                best_estimator.clear_cv_data()

                x_train = normalizer.fit_transform(x_train)
                best_estimator.fit(x_train, y_train, self.data.get_target_ids(train_indices))

                if hasattr(best_estimator, 'feature_importances_'):
                    feature_importances = best_estimator.feature_importances_
                else:
                    feature_importances = []
                time_to_fit = time.time() - t0

                print 'training size: ', training_size, '. best parameter value', ': ', best_params, '. time_to_fit:', time_to_fit
                print 'search time ', search_time

                fold_data = FoldData()
                fold_data.training_size = training_size
                fold_data.estimator = best_estimator
                fold_data.train_inds = train_indices
                fold_data.train_targets = train_targets
                fold_data.time_to_fit = time_to_fit
                fold_data.normalizer = normalizer

                #add grid_search results to main results to be further processed
                self.main_results.add_estimator_results(fold_data,
                                                        trial,
                                                        self.config.trials_per_size,
                                                        feature_importances)

        self.main_results.configs = self.config
        if self.config.save_the_results:
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
                                                   test_target_ids, self.config.cv_loss_function)
        return score

    def _get_best_param_index(self,all_scores):
        mean_scores = []
        for index, scores in enumerate(all_scores):
            mean_scores.append(np.asarray(scores).mean())
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



