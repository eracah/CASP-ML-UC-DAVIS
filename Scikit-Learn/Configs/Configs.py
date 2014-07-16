__author__ = 'Evan Racah'

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from LossFunction import LossFunction
from Estimator import Estimator
from Estimator import GuessEstimator
from Estimator import RankLib
import time
import copy

#Try to import OverrideConfigs.  Used to apply local overrides without modifying this file.
#The goal of this is to prevent accidentally committing changes to this file
imported_override = False
try:
    import OverrideConfigs
    imported_override = True
except:
    pass

class BaseConfigs(object):
    def update_configs(self, **kwargs):
        for key, value in kwargs.iteritems():
            assert hasattr(self, key)
            setattr(self, key, value)

class EstimatorConfigs(BaseConfigs):
    def __init__(self,estimator,params):
        self.estimator = estimator
        self.params = params

    def get_estimator_name(self):
        if isinstance(self.estimator, Estimator):
            return self.estimator.get_name()
        return repr(self.estimator).split('(')[0]

    def get_estimator_short_name(self):
        name = self.get_estimator_name()
        short_name_dict = {
            'KNeighborsRegressor': 'KNR',
            'RandomForestRegressor': 'RFR',
            'RankLib': 'RL',
            'Guess': 'Guess'
        }
        return short_name_dict[name]

    def get_display_name(self, configs):
        name_params = configs.get_name_params()
        for param in name_params:
            value = getattr(configs,param)


class BatchConfigs(BaseConfigs):
    def __init__(self):
        self.all_configs = []

    @staticmethod
    def create_batch_configs():
        batch_configs = BatchConfigs()
        knr_configs = EstimatorConfigs(KNeighborsRegressor(),
                                       {'n_neighbors': [1, 2, 3, 5, 9]})
        rfr_configs = EstimatorConfigs(RandomForestRegressor(n_estimators=16),
                                       {'max_depth': [5, 10, 20, 40, 80, None]})
        rl_configs = EstimatorConfigs(RankLib(),
                                        {'k': [5]})
        guess_configs = EstimatorConfigs(GuessEstimator(),
                                         {'unused': [0]})
        estimator_configs = [rl_configs, knr_configs, rfr_configs, guess_configs]
        for estimator in estimator_configs:
            c = Configs(estimator_configs=estimator)
            batch_configs.all_configs.append(c)
        # batch_configs.vary_config('cv_loss_function',[
        #     LossFunction(LossFunction.MEAN_SQUARED_ERROR),
        #     LossFunction(LossFunction.PRECISION)
        # ])
        return batch_configs

    def vary_config(self, param, values):
        new_configs = []
        for configs in self.all_configs:
            for curr_value in values:
                new_config = copy.deepcopy(configs)
                setattr(new_config, param, curr_value)
                new_config.append(new_config)
        self.all_configs = new_configs

class VisualizationConfigs(BaseConfigs):
    def __init__(self):
        t = time.localtime()
        self.date_string = str(t.tm_mon) + '-' + str(t.tm_mday) + '-' + str(t.tm_year)
        self.show_plots = True
        self.viz_loss_function = LossFunction(LossFunction.PRECISION)
        self.path_to_store_graphs = './Results/Plots/'
        self.show_train = True
        if imported_override:
            OverrideConfigs.apply_viz_overrides(self)
        else:
            print('Couldn''t find OverrideConfigs.py')

class Configs(BaseConfigs):
    def __init__(self, **kwargs):
        self.estimator_configs =  EstimatorConfigs(KNeighborsRegressor(), {'n_neighbors': [1, 2, 3, 7, 13, 21]})
        t = time.localtime()
        self.date_string = str(t.tm_mon) + '-' + str(t.tm_mday) + '-' + str(t.tm_year)
        self.path_to_targets = './Targets/'
        self.path_to_store_results = './Results/Data/'
        self.target_data_file_name = 'SavedData/data.p'
        self.test_size = 0.2
        self.training_sizes = [10, 100, 125, 150, 175, 200, 230]
        self.scoring = 'mean_squared_error'
        self.n_folds = 5
        self.trials_per_size = 5
        self.n_cores = 2
        self.we_learn_the_data = True
        self.show_plots = True
        self.load_target_data = False
        self.save_the_results = True
        self.cv_loss_function = LossFunction(LossFunction.PRECISION)
        self.use_grid_search_cv = False
        self.params_to_vary = []
        #Try to apply local Configs

        self.update_configs(**kwargs)

        if imported_override:
            OverrideConfigs.apply_overrides(self)
        else:
            print('Couldn''t find OverrideConfigs.py')

        self.estimator_name = self.estimator_configs.get_estimator_short_name()
        #after the override so names can be generated from overridden parameters
        self.save_results_file_name = self._generate_save_results_filename()
        self.recall_results_file_name = self.save_results_file_name

    def _get_name_params(self):
        name_params = ['cv_loss_function']
        return name_params

    def _generate_save_results_filename(self):
        # few = 6
        # first_few_letters = slice(0, 6)
        # estimators_used_in_this_run_string = ''.join([estimator_name[first_few_letters] + '_' for estimator_name in self.estimator_names])
        # training_size_range_string = 'train:' + self._get_range_of_values_from_a_list(self.training_sizes)
        #
        # list_of_parameter_name_and_value_strings = []
        # for parameter_name_values_dict in self.parameter_grids:
        #     for parameter_name in parameter_name_values_dict:
        #         parameter_values_list = parameter_name_values_dict[parameter_name]
        #         range_of_parameter_values_string = self._get_range_of_values_from_a_list(parameter_values_list) + '_'
        #         parameter_name_and_value_string = parameter_name[first_few_letters] + delim + range_of_parameter_values_string
        #         list_of_parameter_name_and_value_strings.append(parameter_name_and_value_string)
        #
        # file_string = estimators_used_in_this_run_string + ''.join(list_of_parameter_name_and_value_strings) \
        #               + training_size_range_string + '.dat'
        file_string = self.get_display_name()
        file_string += '.dat'
        return self.date_string + '/' + file_string

    def get_display_name(self):
        delim = ','
        name = self.estimator_name
        name_params = self._get_name_params()
        for param in name_params:
            value = str(getattr(self, param))
            name += delim + param + '=' + value
        return name

    def _get_range_of_values_from_a_list(self, the_list):
        first_number = 0
        last_number = -1
        return str(the_list[first_number]) + '-' + str(the_list[last_number])











