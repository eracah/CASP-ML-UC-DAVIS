__author__ = 'Evan Racah'


import pickle
import numpy as np
import os

#TODO: Have filename be read from a config file
def recall(path, filename):
    with open(path + filename, 'rb') as f:
        return pickle.load(f)


def num_unique(l):
    return len(set(l))

def make_dir_for_file_name(file_name):
    d = os.path.dirname(file_name)
    if not os.path.exists(d):
        os.makedirs(d)

def save_object(object, file_name):
    make_dir_for_file_name(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def check_input(*args):
    for arg in args[1:]:
        assert args[0].shape[0] == arg.shape[0]