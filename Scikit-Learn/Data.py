__author__ = 'Evan Racah'

import glob
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
import random
import os

# Version number to prevent loading out of data data
DATA_VERSION = 1

class TrainTestData(object):
    def __init__(self):
        pass

class Data(object):

    def __init__(self, configs):
        self._version = DATA_VERSION
        self.input_array = np.zeros((0, configs.input_dim))
        self.output_array = np.zeros(0)
        self.target_ids = np.zeros(0,dtype=int)
        self._load_data(configs)
        self.train_target_indices = 0
        (self.train_target_indices, self.test_target_indices) = self._get_test_and_train(configs.test_size)

    def assert_version(self):
        assert self._version == DATA_VERSION

    def is_correct_version(self):
        return self.assert_version()

    def get_targets_array(self):
        num_targets = max(self.target_ids)
        target_array = np.asarray(range(num_targets))
        return target_array

    def select_targets(self, targets):
        to_select = np.ndarray(0, dtype=int)
        for t in targets:
            indices = (self.target_ids == t).nonzero()
            indices = indices[0]
            to_select = np.concatenate((to_select, indices))
        x_targets = self.input_array[to_select, :]
        y_targets = self.output_array[to_select]

        to_select = np.asarray(to_select,dtype=int)
        targets_ids = self.get_target_ids(to_select)
        return x_targets, y_targets, to_select, targets_ids

    def get_test_data(self):
        x_test, y_test, test_inds, test_target_ids = self.select_targets(self.test_target_indices)
        return x_test, y_test, test_inds, test_target_ids

    def sample_train(self, num_train):
        selected_target_ids = random.sample(self.train_target_indices, num_train)
        train_x, train_y, selected_models_indices, _ = self.select_targets(selected_target_ids)
        return train_x, train_y, selected_models_indices, selected_target_ids

    def get_target_ids(self, inds):
        return self.target_ids[inds]

    def get_num_targets(self):
        return self.train_target_indices.max()

    def _load_data(self, configs):
          #get array from csv files
        target_files = glob.glob(configs.path_to_targets + '*.csv')
        if len(target_files) == 0:
            print 'need to unzip first'
            zipCommand = 'unzip ' + configs.path_to_targets + '/' + configs.target_zip_file_name + ' -d ' + configs.path_to_targets
            os.system(zipCommand)
            target_files = glob.glob(configs.path_to_targets + '*.csv')

        for target_id_counter, target_file in enumerate(target_files):
            target = np.genfromtxt(target_file, delimiter=',')
            num_models = target.shape[0]
            self.input_array = np.concatenate((self.input_array, target[:, 0:-1]))
            self.output_array = np.concatenate((self.output_array, target[:, -1]))
            #make a column vector with the target id number
            self.target_ids = np.concatenate((self.target_ids, (target_id_counter + 1) * np.ones(num_models, dtype=int)))
        self.assert_version()


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

    def _get_test_and_train(self, test_size):
        target_array = self.get_targets_array()
        return train_test_split(target_array, test_size=test_size)



