__author__ = 'Evan Racah'

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from LossFunction import LossFunction
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
        self.date_string = str(t.tm_mon) + '-' + str(t.tm_mday) + '-' + str(t.tm_year)
        self.path_to_targets = './Targets/'
        self.path_to_store_results = './Results/Data/'
        self.path_to_store_graphs = './Results/Plots/'
        self.target_data_file_name = 'SavedData/data.p'
        self.test_size = 0.2
        self.training_sizes =[10, 100, 125, 150, 175, 200, 230]
        self.estimators = [KNeighborsRegressor(), RandomForestRegressor()]
        self.estimator_names = [repr(est).split('(')[0] for est in self.estimators]
        self.parameter_grids =[{'n_neighbors': [1, 2, 3, 5, 9]}, {'n_estimators': [1, 2, 4, 8, 16]}]
        self.scoring = 'mean_squared_error'
        self.n_folds = 5
        self.trials_per_size = 5
        self.n_cores = 2
        self.we_learn_the_data = True
        self.show_plots = True
        self.load_target_data = False
        self.save_the_results = True
        self.results_loss_function = LossFunction.MEAN_SQUARED_ERROR
        #Try to apply local Configs
        if imported_override:
            OverrideConfigs.apply_overrides(self)
        else:
            print('Couldn''t find OverrideConfigs.py')

        #after the override so names can be generated from overridden parameters
        self.save_results_file_name = self._generate_save_results_filename()
        self.recall_results_file_name = self.save_results_file_name



    def _generate_save_results_filename(self):
        few = 6
        first_few_letters = slice(0, 6)
        estimators_used_in_this_run_string = ''.join([estimator_name[first_few_letters] + '_' for estimator_name in self.estimator_names])
        training_size_range_string = 'train:' + self._get_range_of_values_from_a_list(self.training_sizes)

        list_of_parameter_name_and_value_strings = []
        for parameter_name_values_dict in self.parameter_grids:
            for parameter_name in parameter_name_values_dict:
                parameter_values_list = parameter_name_values_dict[parameter_name]
                range_of_parameter_values_string = self._get_range_of_values_from_a_list(parameter_values_list) + '_'
                parameter_name_and_value_string = parameter_name[first_few_letters] + ':_' + range_of_parameter_values_string
                list_of_parameter_name_and_value_strings.append(parameter_name_and_value_string)

        file_string = estimators_used_in_this_run_string + ''.join(list_of_parameter_name_and_value_strings) \
                      + training_size_range_string + '.dat'
        return file_string

    def _get_range_of_values_from_a_list(self, the_list):
        first_number = 0
        last_number = -1
        return str(the_list[first_number]) + '-' + str(the_list[last_number])











