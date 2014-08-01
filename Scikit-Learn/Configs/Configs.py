__author__ = 'Evan Racah'

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from LossFunction import LossFunction
from Estimator import Estimator
from Estimator import GuessEstimator
from Estimator import RankLib
from Estimator import ScitkitLearnKNeighborsRegressor
from Estimator import ScitkitLearnRandomForestRegressor
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

    def __init__(self, estimator, params):
        self.estimator = estimator
        self.params = params

    def get_estimator_name(self):
        return self.estimator.get_name()

    @property
    def estimator_short_name(self):
        return self.estimator.get_short_name()

class BatchConfigs(BaseConfigs):
    def __init__(self):
        self.all_configs = []


    @staticmethod
    def create_batch_configs_for_parallelization(num_threads):
        batch_configs = BatchConfigs()
        configs = []
        for n in num_threads:
            estimator_config = EstimatorConfigs(RankLib(),
                                                {
                                                    'leaf': [2, 5, 10],
                                                })
            c = Configs(estimator_configs=estimator_config)
            c.num_cv_processes = n
            configs.append(c)
        batch_configs.all_configs = configs
        return batch_configs

    @staticmethod
    def create_batch_configs():
        batch_configs = BatchConfigs()
        knr_configs = EstimatorConfigs(ScitkitLearnKNeighborsRegressor(),
                                       {
                                           'n_neighbors': [1, 2, 3, 5, 9]
                                       })
        rfr_configs = EstimatorConfigs(ScitkitLearnRandomForestRegressor(),
                                       {
                                           'max_depth': [5, 10, 20, 40, 80, None]
                                       })
        rl_configs = EstimatorConfigs(RankLib(),
                                      {
                                          'leaf': [2, 5, 10],
                                      })
        guess_configs = EstimatorConfigs(GuessEstimator(),
                                         {
                                             'unused': [0]
                                         })
        estimator_configs = [guess_configs, knr_configs, rfr_configs, rl_configs]

        if imported_override:
            estimator_configs = OverrideConfigs.apply_estimator_overrides(estimator_configs)

        else:
            print('Couldn''t find OverrideConfigs.py')

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
                #TODO: do we have to make deep copy here?
                #can't we we just set the attribute in the
                #config
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
        self.show_runtime = False
        if imported_override:
            OverrideConfigs.apply_viz_overrides(self)
        else:
            print('Couldn''t find OverrideConfigs.py')

    def set_runtime_configs(self):
        del self.viz_loss_function
        self.show_train = False
        self.show_runtime = True
        self.y_attribute = 'cv_time'

class Configs(BaseConfigs):
    def __init__(self, **kwargs):
        self.estimator_configs = EstimatorConfigs(ScitkitLearnKNeighborsRegressor(),
                                                   {'n_neighbors': [1, 2, 3, 7, 13, 21]})
        t = time.localtime()
        self.input_dim = 68
        self.date_string = str(t.tm_mon) + '-' + str(t.tm_mday) + '-' + str(t.tm_year)
        self.path_to_targets = './Targets/'
        self.path_to_store_results = './Results/Data/'
        self.target_data_file_name = 'SavedData/data.p'
        self.test_size = 0.2
        self.training_sizes = (10, 100, 125, 150, 175, 200, 230)
        self.scoring = 'mean_squared_error'
        self.n_folds = 5
        self.trials_per_size = 5
        self.n_cores = 2
        self.we_learn_the_data = True
        self.generate_plots = True
        self.show_plots = True
        self.load_target_data = False
        self.save_the_results = True
        self.cv_loss_function = LossFunction(LossFunction.PRECISION)
        self.results_loss_function = LossFunction(LossFunction.PRECISION)
        self.params_to_vary = []
        self.use_ranks_for_y = False
        self.normalize_data = True
        #Try to apply local Configs

        self.update_configs(**kwargs)

        if imported_override:
            OverrideConfigs.apply_overrides(self)
        else:
            print('Couldn''t find OverrideConfigs.py')

        self.estimator_name = self.estimator_configs.estimator_short_name

    @property
    def save_results_file_name(self):
        return self._generate_save_results_filename()

    def _get_params_to_show_in_filename(self):
        name_params = ['cv_loss_function', 'normalize_data']
        return name_params

    def _generate_save_results_filename(self):
        file_string = self.get_display_name()
        file_string += '.dat'
        return self.date_string + '/' + file_string

    def get_display_name(self):
        delim = ','
        name = self.estimator_name
        name_params = self._get_params_to_show_in_filename()
        for param in name_params:
            value = str(getattr(self, param))
            name += delim + param + '=' + value
        return name

    def _get_range_of_values_from_a_list(self, the_list):
        first_number = 0
        last_number = -1
        return str(the_list[first_number]) + '-' + str(the_list[last_number])











