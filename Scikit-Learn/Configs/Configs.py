__author__ = 'Evan Racah'

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import time

#Try to import OverrideConfigs.  Used to apply local overrides without modifying this file.
#The goal of this is to prevent accidentally committing changes to this file
imported_override = False
try:
    import OverrideConfigs
    imported_override = True
except:
    pass

class Configs:
    def __init__(self):
        t = time.localtime()
        date_string = str(t.tm_mon) + '-' + str(t.tm_mday) + '-' + str(t.tm_year)
        self.path_to_targets = './Targets/'
        self.path_to_store_results = './Results/Data/'
        self.path_to_store_graphs = './Results/Plots/'
        self.target_data_file_name = 'SavedData/data.p'
        self.save_results_file_name = date_string + '_Main_Results.dat'
        self.recall_results_file_name = date_string + '_Main_Results.dat'
        self.test_size = 0.2
        self.training_sizes =[10, 100, 125, 150, 175, 200, 230]
        self.estimators = [KNeighborsRegressor(), RandomForestRegressor()]
        self.parameter_grids =[{'n_neighbors': [1, 2, 3, 5, 9]}, {'n_estimators': [1, 2, 4, 8, 16]}]
        self.scoring = 'mean_squared_error'
        self.n_folds = 5
        self.trials_per_size = 5
        self.n_cores = 2
        self.we_learn_the_data = True
        self.show_plots = True
        self.load_target_data = False

        #Try to apply local configs
        if imported_override:
            OverrideConfigs.apply_overrides(self)
        else:
            print('Couldn''t find OverrideConfigs.py')







