__author__ = 'Evan Racah'

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
        self.colors = ['r', 'g', 'y', 'k', 'm', 'c']
        self.color_index = 0
        self.learning_curve_data = Visualization.PlotData()
        self.configs = configs
        self.path = self.configs.path_to_store_graphs
        self.date = self.configs.date_string
        self.prepare_learning_curve_plot()

    def _set_results(self, main_result_obj):
        self.results_obj = main_result_obj
        self.results_obj.configs.results_loss_function = self.configs.viz_loss_function
        self.results_obj.generate_performance_results()

    def plot_all(self):
        self.plot_actual_vs_predicted_curve()

    def new_color(self):
        col = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return col


    def scatter_data(self, estimator_name, sizes, x_name, y_name, line_args, do_scatter_plot=False):
        legend = []
        estimator_results = self.results_obj.estimator_results
        x, y = estimator_results.get_plot_arrays(sizes, (x_name, y_name))
        if do_scatter_plot:
            plt.scatter(x, y, c=self.new_color(), alpha=alpha)
        else:
            # plt.plot(x, y, c=self.new_color(), alpha=alpha, lw=5)
            plt.plot(x, y, **line_args)
        legend.append(estimator_name + ' ' + y_name)
        return legend

    def prepare_learning_curve_plot(self):
        self.learning_curve_data.plot_string = 'Learning_Curve'
        self.learning_curve_data.legend_list = []
        self.learning_curve_data.file_name = self.path + '/' + self.date + '/' + \
                                             self.learning_curve_data.plot_string + '.jpg'
        self.learning_curve_data.x_label = 'Training Size (Number of Targets)'
        self.learning_curve_data.y_label = self.configs.viz_loss_function.get_display_name()
        plt.figure(self.fig_number)

    def add_to_learning_curve_plot(self, main_results):
        self._set_results(main_results)
        estimator_name = self.results_obj.estimator_name
        training_sizes = self.results_obj.configs.training_sizes

        line_args = {
            'c' : self.new_color(),
            'alpha' : 1,
            'lw' : 2,
            'linestyle' : 'solid',
            'marker' : 'o'
        }
        l1 = self.scatter_data(estimator_name, training_sizes, 'training_size', 'test_error', line_args)
        line_args['linestyle'] = 'dashed'
        l2 = self.scatter_data(estimator_name, training_sizes, 'training_size', 'train_error', line_args)
        self.learning_curve_data.legend_list += l1 + l2

    def finish_learning_curve_plot(self):
        self.set_plot_captions(self.learning_curve_data.plot_string,
                               self.learning_curve_data.x_label,
                               self.learning_curve_data.y_label,
                               legend_list=self.learning_curve_data.legend_list)

        make_dir_for_file_name(self.learning_curve_data.file_name)
        plt.savefig(self.learning_curve_data.file_name)

        self.fig_number += 1


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
