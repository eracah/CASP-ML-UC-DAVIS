__author__ = 'Evan Racah'

import glob
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import time

#get list of target files
path_to_targets = './Targets/'
path_to_store_results = './Results/Data/'
path_to_store_graphs = './Results/Plots/'


test_size = 0.2
training_sizes =[10,
                 20,
                 50]
                 # 100,
                 # 150,
                 # 200,
                 # 230]

estimators = [KNeighborsRegressor()], RandomForestRegressor()]
parameter_grids =[{'n_neighbors': [1, 2, 3, 5, 9]}]#, {'n_estimators': [1, 2, 4, 8, 16]}]
scoring = 'mean_squared_error'
n_folds = 5
trials_per_size = 2
n_cores = 2

t = time.localtime()
date_string = str(t.tm_mon)+ '-' + str(t.tm_mday) + '-' + str(t.tm_year)

save_results_file_name = date_string + '_Main_Results.dat'
recall_results_file_name = date_string + '_Main_Results.dat'
