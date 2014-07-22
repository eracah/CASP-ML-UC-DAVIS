__author__ = 'Evan Racah'

from sklearn import metrics

import pickle
import numpy as np
import os
import time
from math import ceil
from Configs.Configs import Configs
from LossFunction import LossFunction
from HelperFunctions import save_object
from Estimator import Estimator
from sklearn.preprocessing import StandardScaler


class FoldData(object):
    pass

class TrainingSampleResult(object):

    #TODO: add in more time metrics to this class
    def __init__(self, trials_per_size):
        self.configs = Configs()
        self.fold_data = []
        self.trials = trials_per_size
        self.count = 0
        self.test_error = 0
        self.train_error = 0

    def get_fold_data_attributes(self, attribute):
        return [getattr(data,attribute) for data in self.fold_data]

    def add_sample_results(self, fold_data, data):
        self.fold_data.append(fold_data)
        self._generate_predictions(data, self.count)
        self.count += 1

    def generate_performance_results(self, data, configs):
        num_folds = len(self.fold_data)
        num_pred = len(self.fold_data[0].test_predicted_values)
        num_actual = len(self.fold_data[0].test_actual_values)
        train_perf = np.ndarray(num_folds)
        test_perf = np.ndarray(num_folds)
        test_predicted = np.ones((num_folds,num_pred,))
        test_actual = np.ones((num_folds,num_actual,))
        for index, fold in enumerate(self.fold_data):
            # train_targets = self.train_targets[index]
            # test_targets = self.test_targets[index]
            train_predicted = fold.train_predicted_values
            train_actual = fold.train_actual_values
            test_predicted[index] = fold.test_predicted_values
            test_actual[index] = fold.test_actual_values
            results_loss_function = configs.results_loss_function
            train_target_ids = data.get_target_ids(fold.train_inds)
            test_target_ids = data.get_target_ids(fold.test_inds)
            train_perf[index] = LossFunction.compute_loss_function(train_predicted,
                                                                   train_actual,
                                                                   train_target_ids,
                                                                   results_loss_function)
            test_perf[index] = LossFunction.compute_loss_function(test_predicted[index],
                                                                  test_actual[index],
                                                                  test_target_ids,
                                                                  results_loss_function)
            self.fold_data[index].train_error = train_perf[index]
            self.fold_data[index].test_error = test_perf[index]
        self.test_actual = test_actual
        self.train_actual = train_actual
        self.test_predicted = test_predicted
        self.train_predicted = train_predicted
        # self.train_error = train_perf
        # self.test_error = test_perf

    def _generate_predictions(self, data, index):
        x_test, y_test, test_inds, test_target_ids = data.get_test_data()
        fold = self.fold_data[index]
        train_targets = fold.train_targets
        x_train, y_train, _, train_target_ids = data.select_targets(train_targets)

        x_train = fold.normalizer.transform(x_train)
        x_test = fold.normalizer.transform(x_test)


        train_predict_start_time = time.time()
        fold.train_predicted_values = fold.estimator.predict(x_train, y_train, train_target_ids)
        fold.train_predict_time = time.time() - train_predict_start_time
        test_predict_start_time = time.time()
        fold.test_predicted_values = fold.estimator.predict(x_test, y_test, test_target_ids)
        fold.test_predict_time = time.time() - test_predict_start_time
        fold.train_actual_values = y_train
        fold.test_actual_values = y_test
        fold.test_inds = test_inds
        print '\ttrain predict time: ', fold.train_predict_time
        print '\ttest predict time: ', fold.test_predict_time
        # Hacky solution to result file size - the grid search object is very large
        # Better to just not make estimator an attribute in the first place
        del fold.estimator
        

class EstimatorResult(object):

    def __init__(self, estimator_name):
        self.name = estimator_name
        self.training_sample_dict = {}

    def add_training_results(self, training_size, fold_data, trial,
                             trials_per_size, data):
        if trial == 0:
            self.training_sample_dict[training_size] = TrainingSampleResult(trials_per_size)

        self.training_sample_dict[training_size].add_sample_results(fold_data,
                                                                    data)


    def get_aggregated_data(self, data_name, sizes):
        training_sample_results = [self.training_sample_dict[size] for size in sizes]
        #must have axis=0 so we average arrays of arrays and get out an array instead of one value
        all_data = [r.get_fold_data_attributes(data_name) for r in training_sample_results]
        all_data = [np.asarray(sublist) for sublist in all_data]
        means = [obj.mean(axis=0) for obj in all_data]
        variances = [obj.var() for obj in all_data]
        return means, variances

    def get_plot_arrays(self, sizes, names):
        ret = len(names)*[0]
        all_vars = []
        for i, name in enumerate(names):
            if name == 'training_size':
                ret[i] = sizes
            else:
                means, vars = self.get_aggregated_data(name, sizes)
                ret[i] = means
                all_vars.append(vars)
        return ret, all_vars
    
    def generate_performance_results(self, data, configs):
        for training_sample_results in self.training_sample_dict.itervalues():
           training_sample_results.generate_performance_results(data, configs)

class MainResult(object):


    def __init__(self, estimator_name, path_to_store_results, file_name):
        self.feature_importances = np.ndarray(0,dtype=float)
        self.estimator_name = estimator_name

        self.estimator_results = EstimatorResult(estimator_name)

        self.filename = file_name
        self.path = path_to_store_results

    def generate_performance_results(self):
        data = self.data
        self.estimator_results.generate_performance_results(data, self.configs)

    def get_mean_feature_importance(self):
        print self.feature_importances
        return np.mean(self.feature_importances, axis=0)
    def add_estimator_results(self, fold_data, trial, trials_per_size):
        self.estimator_results.add_training_results(fold_data.training_size,
                                                    fold_data,
                                                    trial,
                                                    trials_per_size,
                                                    self.data)

    #TODO: Move this outside of this class
    def save_data(self):
        file_name = self.path + self.filename
        save_object(self, file_name)













