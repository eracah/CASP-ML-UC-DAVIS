__author__ = 'Evan Racah'

import random
import time
import numpy as np
import glob
import pickle
import os.path
import copy
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from Results import MainResult
from LossFunction import LossFunction
from Estimator import Estimator
import scipy.stats as ss


# Version number to prevent loading out of data data
DATA_VERSION = 1

class Data(object):
    def __init__(self):
        self._version = DATA_VERSION
        self.use_ranks_for_y = False
        pass

    def assert_version(self):
        assert self._version == DATA_VERSION

    def is_correct_version(self):
        return self._version == DATA_VERSION

    def get_targets_array(self):
        num_targets = max(self.target_ids)
        target_array = np.asarray(range(num_targets))
        return target_array

    def select_targets(self, targets):
        to_select = np.ndarray(0, dtype=int)
        for t in targets:
            inds = (self.target_ids == t).nonzero()
            inds = inds[0]
            to_select = np.concatenate((to_select, inds))
        x_targets = self.input_array[to_select, :]
        if self.use_ranks_for_y:
            y_targets = self.rank_output_array[to_select, :]
        else:
            y_targets = self.output_array[to_select]

        to_select = np.asarray(to_select,dtype=int)
        targets_ids = self.get_target_ids(to_select)
        return x_targets, y_targets, to_select, targets_ids

    def sample_train(self, num_train):
        train_targets = random.sample(self.train_targets, num_train)
        train_x, train_y, selected_inds, _ = self.select_targets(train_targets)

        #TODO: Normalize train_x here to prevent data snooping
        # normalizes every column
        # train_x = stats.zscore(train_x, axis=0)
        return train_x, train_y, selected_inds, train_targets

    def get_target_ids(self, inds):
        return self.target_ids[inds]

    def get_num_targets(self):
        return self.train_targets.max()

    @staticmethod
    def make_score_0_1(Y, target_ids, k):
        target_ids_set = set(target_ids.tolist())
        Y_0_1 = Y.copy()
        for index, target_id in enumerate(target_ids_set):
            target_inds = (target_ids == target_id).nonzero()
            Y_target = Y[target_inds]
            top_k_inds = Y_target.argsort()[:-k - 1:-1]
            relevance = np.zeros(len(Y_target))
            relevance[top_k_inds] = 1
            Y_0_1[target_inds] = relevance
        return Y_0_1

class Learn():
    def __init__(self, config):
        self.config = config

        data = self._get_data(self.config.path_to_targets,self.config)

        self.data = self._get_test_and_train(data, config.test_size)
        if self.config.use_ranks_for_y:
            self.data.use_ranks_for_y = True
        self.estimator_configs = config.estimator_configs
        self.estimator_name = config.estimator_name
        # instantiate a main result object
        self.main_results = MainResult(self.estimator_name, self.config.path_to_store_results,
                                       self.config.save_results_file_name)



    def run_grid_search(self):
        self.main_results.data = self.data
        for training_size in self.config.training_sizes:
            for trial in range(self.config.trials_per_size):
                x_train, y_train, train_inds, train_targets = self.data.sample_train(training_size)

                # for each estimator (ML technique)
                estimator_configs = self.estimator_configs

                num_folds = self.config.n_folds
                t0 = time.time()

                if not self.config.use_grid_search_cv:
                    sampled_targets = np.asarray(list(set(train_targets)),dtype=int)
                    kf = cross_validation.KFold(len(sampled_targets),num_folds,shuffle=True)

                    params = estimator_configs.params
                    param_grid = list(ParameterGrid(params))
                    cv_scores = []
                    for i in range(len(param_grid)):
                        cv_scores.append([])

                    for train_target_indices, test_target_indices in kf:
                        cv_train_targets = sampled_targets[train_target_indices]
                        cv_test_targets = sampled_targets[test_target_indices]

                        cv_train_x, cv_train_y, _, cv_train_target_ids = self.data.select_targets(cv_train_targets)
                        cv_test_x, cv_test_y, _, cv_test_target_ids = self.data.select_targets(cv_test_targets)

                        for param_index, params in enumerate(param_grid):
                            estimator_configs.estimator.set_params(**params)
                            if isinstance(estimator_configs.estimator,Estimator):
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
                                estimator_configs.estimator.fit(cv_train_x,cv_train_y)
                                cv_test_pred = estimator_configs.estimator.predict(cv_test_x)
                            score = LossFunction.compute_loss_function(cv_test_pred, cv_test_y,
                                                                       cv_test_target_ids, self.config.cv_loss_function)
                            cv_scores[param_index].append(score)
                    best_param_index = self._get_best_param_index(cv_scores)
                    best_params = param_grid[best_param_index]
                    best_estimator = copy.deepcopy(estimator_configs.estimator)
                    best_estimator.set_params(**best_params)
                else:
                    grid_search = GridSearchCV(estimator_configs.estimator, estimator_configs.params,
                                               scoring=self.config.scoring, cv=self.config.n_folds,
                                               n_jobs=self.config.n_cores)


                    #find best fit for training data
                    grid_search.fit(x_train, y_train)
                    best_params = grid_search.best_params_

                search_time = time.time() - t0

                t0 = time.time()
                #fit estimator with best parameters to training data
                if not self.config.use_grid_search_cv:
                    if isinstance(best_estimator, Estimator):
                        best_estimator.clear_cv_data()
                        best_estimator.fit(x_train, y_train, self.data.get_target_ids(train_inds))
                    else:
                        best_estimator.fit(x_train, y_train)
                else:
                    grid_search.best_estimator_.fit(x_train, y_train)
                    best_estimator = grid_search.best_estimator_

                if hasattr(best_estimator, 'feature_importances_'):
                    feature_importances = best_estimator.feature_importances_
                else:
                    feature_importances = []
                time_to_fit = time.time() - t0


                print 'training size: ', training_size, '. best parameter value', ': ', best_params, '. time_to_fit:', time_to_fit
                print 'search time ', search_time



                #add grid_search results to main results to be further processed
                self.main_results.add_estimator_results(self.estimator_name,
                                                        training_size,
                                                        best_estimator,
                                                        train_inds,
                                                        train_targets,
                                                        time_to_fit,
                                                        trial,
                                                        self.config.trials_per_size,
                                                        feature_importances)



        self.main_results.configs = self.config
        if self.config.save_the_results:
            self.main_results.save_data()

        return self.main_results
    def _get_best_param_index(self,all_scores):
        mean_scores = []
        for index, scores in enumerate(all_scores):
            mean_scores.append(np.asarray(scores).mean())
        return np.asarray(mean_scores).argmin()
    @staticmethod
    def _get_test_and_train(data, test_size):
        target_array = data.get_targets_array()
        data.train_targets, data.test_targets = train_test_split(target_array, test_size=test_size)

        return data

    @staticmethod
    def _get_data(path,configs):
        data_file_name = configs.target_data_file_name
        load_target_data = configs.load_target_data
        data = []
        file_exists = os.path.isfile(data_file_name)
        if load_target_data and file_exists:
            with open(data_file_name, 'rb') as f:
                data = pickle.load(f)
        if not load_target_data or not file_exists or not data.is_correct_version():
            # convert every target csv file to numpy matrix and split up between
            #input and output (label)
            target_files = glob.glob(path + '*.csv')
            data = Data()
            input_dim = 68
            target_id_counter = 1
            input_array = np.zeros((0, input_dim))
            output_array = np.zeros(0)
            rank_output_array = np.zeros(0)
            target_ids = np.zeros(0,dtype=int)
            for target_file in target_files:
                #get array from csv files
                target = np.genfromtxt(target_file, delimiter=',')

                num_models = target.shape[0]
                input_array = np.concatenate((input_array, target[:, 0:-1]))
                output_array = np.concatenate((output_array, target[:, -1]))
                rank_output_array = np.concatenate((rank_output_array,ss.rankdata(target[:, -1])))
                #make a column vector with the target id number
                target_ids = np.concatenate((target_ids, target_id_counter * np.ones(num_models,dtype=int)))
                target_id_counter += 1
            data.input_array = input_array
            data.output_array = output_array
            data.rank_output_array = rank_output_array
            data.target_ids = target_ids

            with open(data_file_name, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        data.assert_version()
        return data




