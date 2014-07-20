__author__ = 'Evan Racah'

import glob
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Version number to prevent loading out of data data
DATA_VERSION = 1


class Data(object):

    def __init__(self, configs):
        self._version = DATA_VERSION
        self.input_array = np.zeros((0, configs.input_dim))
        self.output_array = np.zeros(0)
        self.target_ids = np.zeros(0,dtype=int)
        self.normalizer = StandardScaler()

        self._load_data(configs)
        self.train_target_indices, self.test_target_indices = self._get_test_and_train(configs.test_size)

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
        #normalize x using same mean and variance from x_train
        x_test = self.normalizer.transform(x_test)
        return x_test, y_test, test_inds, test_target_ids

    def sample_train(self, num_train):
        train_targets = random.sample(self.train_target_indices, num_train)
        train_x, train_y, selected_inds, _ = self.select_targets(train_targets)

        #normalize x-train
        self.normalizer = self.normalizer.fit(train_x)
        train_x = self.normalizer.transform(train_x)
        return train_x, train_y, selected_inds, train_targets

    def get_target_ids(self, inds):
        return self.target_ids[inds]

    def get_num_targets(self):
        return self.train_target_indices.max()

    def _load_data(self, configs):
        target_files = glob.glob(configs.path_to_targets + '*.csv')
        for target_id_counter, target_file in enumerate(target_files):
            #get array from csv files
            target = np.genfromtxt(target_file, delimiter=',')
            num_models = target.shape[0]
            self.input_array = np.concatenate((self.input_array, target[:, 0:-1]))
            self.output_array = np.concatenate((self.output_array, target[:, -1]))
            #make a column vector with the target id number
            self.target_ids = np.concatenate((self.target_ids, (target_id_counter + 1) * np.ones(num_models, dtype=int)))


        with open(configs.target_data_file_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
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



