__author__ = 'Evan Racah'

import time
import random
import numpy as np
import glob
import pickle
import os.path
import configs
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from Results import MainResult



class Data:
    def __init__(self):
        pass

    def get_targets_array(self):
        num_targets = max(self.target_ids)
        target_array = np.asarray(range(num_targets))
        return target_array

    def select_targets(self, targets):
        to_select = np.ndarray(0, dtype=int)
        for t in targets:
            inds, unused = (self.target_ids == t).nonzero()
            to_select = np.concatenate((to_select, inds))
        x_targets = self.input_array[to_select, :]
        y_targets = self.output_array[to_select]
        return x_targets, y_targets, np.asarray(to_select,dtype=int)

    def sample_train(self, num_train):
        train_targets = random.sample(self.train_targets, num_train)
        train_x, train_y, selected_inds = self.select_targets(train_targets)
        return train_x, train_y, selected_inds, train_targets


class Learn():
    def __init__(self, config):
        self.config = config

        data = self._get_data(self.config.path_to_targets)

        self.data = self._get_test_and_train(data, config.test_size)
        self.estimators = config.estimators

        # take the 'estimator_name' from 'estimator_name()' ie 'KNeighborsRegressor()' becomes 'KNeighborsRegressor'
        self.estimator_names = [repr(est).split('(')[0] for est in self.estimators]

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
                    grid_search = GridSearchCV(estimator, self.config.parameter_grids[index],
                                               scoring=self.config.scoring, cv=self.config.n_folds,
                                               n_jobs=self.config.n_cores)

                    t0 = time.time()
                    #find best fit for training data
                    grid_search.fit(x_train, y_train)
                    search_time = time.time() - t0

                    t0 = time.time()
                    #fit estimator with best parameters to training data
                    grid_search.best_estimator_.fit(x_train, y_train)
                    time_to_fit = time.time() - t0

                    print 'training size: ', training_size, '. best parameter value', ': ', grid_search.best_params_, '. time_to_fit:', time_to_fit
                    print 'search time ', search_time


                    #TODO call add estimator results outside this loop in the training size loop not in the trial loop
                    #TODO in the trial loop, make an array of grid search objects and x_train, y_train objects, and pass that to the add estimator_results
                    #add grid_search results to main results to be further processed
                    self.main_results.add_estimator_results(self.estimator_names[index],
                                                            training_size,
                                                            grid_search,
                                                            train_inds,
                                                            train_targets,
                                                            time_to_fit,
                                                            trial,
                                                            self.config.trials_per_size)
        self.main_results.save_data()
        return self.main_results

    @staticmethod
    def _get_test_and_train(data, test_size):
        target_array = data.get_targets_array()
        data.train_targets, data.test_targets = train_test_split(target_array, test_size=test_size)
        return data

    @staticmethod
    def _get_data(path):
        data_file_name = configs.target_data_file_name
        load_target_data = configs.load_target_data
        if load_target_data and os.path.isfile(data_file_name):
            data = pickle.load(open(data_file_name, 'rb'))
        else:
            # convert every target csv file to numpy matrix and split up between
            #input and output (label)
            target_files = glob.glob(path + '*.csv')
            # preallocate python lists
            data = Data()
            input_dim = 68
            target_id_counter = 0
            input_array = np.zeros((0, input_dim))
            output_array = np.zeros(0)
            target_ids = np.zeros((0, 1))
            for target_file in target_files:
                # all target files of form target_number.csv, so
                #this strips the .csv to get the number as a string and then makes it an int
                target_index = int(target_file[len(path):-len('.csv')])

                #get array from csv files
                target = np.genfromtxt(target_file, delimiter=',')

                num_models = target.shape[0]
                input_array = np.concatenate((input_array, target[:, 0:-1]))
                output_array = np.concatenate((output_array, target_id_counter * target[:, -1]))
                target_ids = np.concatenate((target_ids, target_id_counter * np.ones((num_models, 1))))
                target_id_counter += 1
            data.input_array = input_array
            data.output_array = output_array
            data.target_ids = target_ids
            pickle.dump(data, open(data_file_name, 'wb'),pickle.HIGHEST_PROTOCOL)
        return data



