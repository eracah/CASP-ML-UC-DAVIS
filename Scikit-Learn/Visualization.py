__author__ = 'Evan Racah'

from os import system

try:
    from matplotlib import pyplot as plt
except ImportError:
    system('module load python matplotlib')
    from matplotlib import pyplot as plt

import math
import numpy as np
from LossFunction import LossFunction
from HelperFunctions import make_dir_for_file_name
class Visualization(object):

    class PlotData(object):
        def __init__(self):
            pass

    def __init__(self, configs):
        self.fig_number = 1
        self.colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c']
        self.color_index = 0
        self.plot_data = Visualization.PlotData()
        self.configs = configs
        self.path = self.configs.path_to_store_graphs
        self.date = self.configs.date_string

        if configs.show_runtime:
            plot_title = 'Runtime'
            y_label = self.configs.y_attribute
        else:
            plot_title = 'Learning_Curve'
            y_label = self.configs.viz_loss_function.get_display_name()
        self.prepare_2d_plot(plot_title, y_label)

    def _set_results(self, main_result_obj):
        self.results_obj = main_result_obj
        if hasattr(self.configs,'viz_loss_function'):
            #TODO: PLA put this setting of results loss function into configs
            # self.results_obj.configs.results_loss_function = self.configs.viz_loss_function
            self.results_obj.generate_performance_results()

    def plot_all(self):
        pass
        #self.plot_actual_vs_predicted_curve()

    def new_color(self):
        col = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return col


    def scatter_data(self, estimator_name, sizes, x_name, y_name, line_args, do_scatter_plot=False):
        legend = []
        estimator_results = self.results_obj.estimator_results
        arrays, all_vars = estimator_results.get_plot_arrays(sizes, (x_name, y_name))
        x = arrays[0]
        y = arrays[1]
        y_vars = all_vars[0]
        if do_scatter_plot:
            plt.scatter(x, y, c=self.new_color(), alpha=alpha)
        else:
            plt.errorbar(x, y, yerr=y_vars, **line_args)
        legend.append(estimator_name + ' ' + y_name)
        return legend

    def prepare_2d_plot(self, plot_title, y_label):
        self.plot_data.plot_string = plot_title
        self.plot_data.legend_list = []
        self.plot_data.file_name = self.path + '/' + self.date + '/' + \
                                   self.plot_data.plot_string + '.jpg'
        self.plot_data.x_label = 'Training Size (Number of Targets)'
        self.plot_data.y_label = y_label
        plt.figure(self.fig_number)

    def add_to_learning_curve_plot(self, main_results):
        self._set_results(main_results)
        estimator_name = self.results_obj.configs.estimator_configs.estimator_short_name
        training_sizes = self.results_obj.configs.training_sizes

        train_line_style = 'dashed'
        test_line_style = 'solid'
        line_args = {
            'c': self.new_color(),
            'alpha': 1,
            'lw': 2,
            'linestyle': test_line_style,
            'capsize': 20
        }
        if self.configs.show_runtime:
            y_attributes = [self.configs.y_attribute, self.configs.y_attribute]
        else:
            y_attributes = ['test_error', 'train_error']
        l1 = self.scatter_data(estimator_name, training_sizes, 'training_size', y_attributes[0], line_args)
        self.plot_data.legend_list += l1
        line_args['linestyle'] = train_line_style
        if self.configs.show_train:
            l2 = self.scatter_data(estimator_name, training_sizes, 'training_size', y_attributes[1], line_args)
            self.plot_data.legend_list += l2

    def finish_learning_curve_plot(self):
        self.set_plot_captions(self.plot_data.plot_string,
                               self.plot_data.x_label,
                               self.plot_data.y_label,
                               legend_list=self.plot_data.legend_list)

        if self.configs.show_runtime:
            plt.ylim([0, 300])
        make_dir_for_file_name(self.plot_data.file_name)
        plt.savefig(self.plot_data.file_name)

        self.fig_number += 1


    # def plot_feature_importance_bar_chart(self):
    #     mean_importances = self.results_obj.get_mean_feature_importance()
    #     print mean_importances.shape
    #     print len(mean_importances)
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     ax.bar(range(), mean_importances, width=1)

    def plot_actual_vs_predicted_curve(self):
        #TODO: This doesn't work anymore.
        plot_string = 'Predicted_vs_Actual_Scatter'
        sizes = self.results_obj.configs.training_sizes
        estimator_name = self.results_obj.estimator_name

        line_args = {
            'c' : self.new_color(),
            'alpha' : 1,
            'marker' : 'o'
        }

        plt.figure(self.fig_number)
        for i, size in enumerate(sizes):

            rows_of_subplot = len(sizes)
            plt.subplot(rows_of_subplot*100 + 4*10 + i)
            self.scatter_data(estimator_name, [size], 'test_actual', 'test_predicted',
                              line_args, do_scatter_plot=True)

            x = np.linspace(0.0, 1.0, 1000)
            plt.plot(x, x, color='b',linestyle='--')

            title_string = estimator_name[0:3] + ' and size of ' + str(size)
            self.set_plot_captions(title_string, 'Actual Value', 'Predicted Value')
        file_name = self.path + '/' + self.date + '/' + estimator_name + '_' + plot_string + '.jpg'
        make_dir_for_file_name(file_name)
        plt.savefig(file_name)
        self.fig_number += 1


    def plot_parameter_error(self):
        pass

    def show(self):
        plt.show()


    def set_plot_captions(self, plot_title, x_label, y_label, legend_list=''):
        plt.legend(legend_list)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(plot_title)
