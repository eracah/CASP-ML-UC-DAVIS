__author__ = 'Evan Racah'

import glob
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

#get list of target files
path = './Targets/'


test_size = 0.2
training_sizes =[10,
                 20,
                 50,
                 100,
                 150,
                 200,
                 230]
estimators = [KNeighborsRegressor(), RandomForestRegressor()]
parameter_grids =[{'n_neighbors': [1, 2, 3, 5, 9]}, {'n_estimators': [1, 2, 4, 8, 16 ]}]
scoring = 'mean_squared_error'
n_folds = 5
trials_per_size = 5
