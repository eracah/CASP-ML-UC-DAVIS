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


# Version number to prevent loading out of data data
DATA_VERSION = 1

class Data:
    def __init__(self):
        self._version = DATA_VERSION
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
        y_targets = self.output_array[to_select]
        to_select = np.asarray(to_select,dtype=int)
        targets_ids = self.get_target_ids(to_select)
        return x_targets, y_targets, to_select, targets_ids

    def sample_train(self, num_train):
        train_targets = random.sample(self.train_targets, num_train)
        train_x, train_y, selected_inds, _ = self.select_targets(train_targets)
        return train_x, train_y, selected_inds, train_targets

    def get_target_ids(self, inds):
        return self.target_ids[inds]

    def get_num_targets(self):
        return self.train_targets.max()


class Learn():
    def __init__(self, config):
        self.config = config

        data = self._get_data(self.config.path_to_targets,self.config)
        self.data = self._get_test_and_train(data, config.test_size)
        self.estimators = config.estimators
        self.estimator_names = config.estimator_names
        # instantiate a main result object
        self.main_results = MainResult(self.estimator_names, self.config.path_to_store_results,
                                       self.config.save_results_file_name)


    def run_grid_search(self):
        self.main_results.data = self.data
        for training_size in self.config.training_sizes:
            for trial in range(self.config.trials_per_size):
                x_train, y_train, train_inds, train_targets = self.data.sample_train(training_size)

                # for each estimator (ML technique)
                for index, estimator in enumerate(self.estimators):
                    #instantiate grid search object
                    num_folds = self.config.n_folds
                    t0 = time.time()

                    if not self.config.use_grid_search_cv:
                        sampled_targets = np.asarray(list(set(train_targets)),dtype=int)
                        kf = cross_validation.KFold(len(sampled_targets),num_folds,shuffle=True)

                        params = self.config.parameter_grids[index]
                        param_grid = list(ParameterGrid(params))
                        cv_scores = []
                        for i in range(len(param_grid)):
                            cv_scores.append([])

                        for train_target_indices, test_target_indices in kf:
                            cv_train_targets = sampled_targets[train_target_indices]
                            cv_test_targets = sampled_targets[test_target_indices]

                            cv_train_x, cv_train_y, _, _ = self.data.select_targets(cv_train_targets)
                            cv_test_x, cv_test_y, _, cv_test_target_ids = self.data.select_targets(cv_test_targets)

                            for param_index, params in enumerate(param_grid):
                                estimator.set_params(**params)
                                estimator.fit(cv_train_x,cv_train_y)
                                cv_test_pred = estimator.predict(cv_test_x)
                                score = LossFunction.compute_loss_function(cv_test_pred, cv_test_y,
                                                                           cv_test_target_ids, self.config.cv_loss_function)
                                cv_scores[param_index].append(score)
                                pass
                        best_param_index = self._get_best_param_index(cv_scores)
                        best_params = param_grid[best_param_index]
                        best_estimator = copy.deepcopy(estimator)
                        best_estimator.set_params(**best_params)
                    else:
                        grid_search = GridSearchCV(estimator, self.config.parameter_grids[index],
                                                   scoring=self.config.scoring, cv=self.config.n_folds,
                                                   n_jobs=self.config.n_cores)


                        #find best fit for training data
                        grid_search.fit(x_train, y_train)
                        best_params = grid_search.best_params_

                    search_time = time.time() - t0

                    t0 = time.time()
                    #fit estimator with best parameters to training data
                    if not self.config.use_grid_search_cv:
                        best_estimator.fit(x_train,y_train)
                    else:
                        grid_search.best_estimator_.fit(x_train, y_train)
                        best_estimator = grid_search.best_estimator_

                    time_to_fit = time.time() - t0

                    print 'training size: ', training_size, '. best parameter value', ': ', best_params, '. time_to_fit:', time_to_fit
                    print 'search time ', search_time



                    #add grid_search results to main results to be further processed
                    self.main_results.add_estimator_results(self.estimator_names[index],
                                                            training_size,
                                                            best_estimator,
                                                            train_inds,
                                                            train_targets,
                                                            time_to_fit,
                                                            trial,
                                                            self.config.trials_per_size)



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
            # preallocate python lists
            data = Data()
            input_dim = 68
            target_id_counter = 1
            input_array = np.zeros((0, input_dim))
            output_array = np.zeros(0)
            target_ids = np.zeros(0,dtype=int)
            for target_file in target_files:
                # all target files of form target_number.csv, so
                #this strips the .csv to get the number as a string and then makes it an int
                target_index = int(target_file[len(path):-len('.csv')])

                #get array from csv files
                target = np.genfromtxt(target_file, delimiter=',')

                num_models = target.shape[0]
                input_array = np.concatenate((input_array, target[:, 0:-1]))
                output_array = np.concatenate((output_array, target[:, -1]))
                target_ids = np.concatenate((target_ids, target_id_counter * np.ones(num_models,dtype=int)))
                target_id_counter += 1
            data.input_array = input_array
            data.output_array = output_array
            data.target_ids = target_ids
            with open(data_file_name, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        data.assert_version()
        return data



