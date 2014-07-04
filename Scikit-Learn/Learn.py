__author__ = 'Evan Racah'

import time
import random
import numpy as np
import glob
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from Results import MainResult
from HelperFunctions import concatenate_arrays


class Learn():

    def __init__(self, config):
        self.config = config

        input_matrices, output_matrices = self._get_data(self.config.path_to_targets)
        self.x_test, self.y_test, self.x_total_train, self.y_total_train = self._get_test_and_train(input_matrices,
                                                                                                    output_matrices,
                                                                                                    config.test_size)
        self.estimators = config.estimators

        #take the 'estimator_name' from 'estimator_name()' ie 'KNeighborsRegressor()' becomes 'KNeighborsRegressor'
        self.estimator_names = [repr(est).split('(')[0] for est in self.estimators]

        #instantiate a main result object
        self.main_results = MainResult(self.estimator_names, self.config.path_to_store_results, self.config.save_results_file_name)


    def run_grid_search(self):

        for training_size in self.config.training_sizes:
            for trial in range(self.config.trials_per_size):
                #get correctly sized subset of training data by
                #sampling a a certain number of indices from all the possible
                #target indices
                train_indices = random.sample(range(len(self.x_total_train)), training_size)

                #put all targets selected above into one array per x and y
                x_train, y_train = concatenate_arrays(train_indices, [self.x_total_train, self.y_total_train])

                #for each estimator (ML technique)
                for index, estimator in enumerate(self.estimators):
                    #instantiate grid search object
                    grid_search = GridSearchCV(estimator, self.config.parameter_grids[index], scoring=self.config.scoring, cv=self.config.n_folds, n_jobs=self.config.n_cores)

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

                    #add grid_search results to main results to be further processed
                    self.main_results.add_estimator_results(self.estimator_names[index], training_size, grid_search, (self.x_test, self.y_test), (x_train, y_train),
                                                       time_to_fit, trial, self.config.trials_per_size)
        return self.main_results

    @staticmethod
    def _get_test_and_train(input_matrices, output_matrices, test_size):
        #split up into training and testing using test_size as the proportion value
        x_total_train, x_total_test, y_total_train, y_total_test = train_test_split(input_matrices, output_matrices,
                                                                                    test_size=test_size)
        #concatenate all the targets from test together into one big array (x for features, y for labels)
        x_test, y_test = concatenate_arrays(range(len(x_total_test)), [x_total_test, y_total_test])
        return x_test, y_test, x_total_train, y_total_train

    @staticmethod
    def _get_data(path):

        target_files = glob.glob(path + '*.csv')
        #preallocate python lists
        input_matrices = len(target_files)*[0]
        output_matrices = len(target_files)*[0]

        #convert every target csv file to numpy matrix and split up between
        #input and output (label)
        for target_file in target_files:
            # all target files of form target_number.csv, so
            #this strips the .csv to get the number as a string and then makes it an int
            target_index = int(target_file[len(path):-len('.csv')])

            #get array from csv files
            target = np.genfromtxt(target_file, delimiter=',')

            #last column of target array is the output (gdt-ts values)
            output_matrices[target_index-1] = target[:, -1]

            #the rest of array is all the features (the input)
            input_matrices[target_index-1] = target[:, 0:-1]

        return np.asarray(input_matrices), np.asarray(output_matrices)



