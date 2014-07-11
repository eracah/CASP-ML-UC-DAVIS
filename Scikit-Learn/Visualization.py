__author__ = 'Evan Racah'

from matplotlib import pyplot as plt
import math
import numpy as np
from LossFunction import LossFunction
class Visualization(object):

    def __init__(self, main_result_obj, configs):
        self.fig_number = 1
        self.results_obj = main_result_obj
        self.configs = configs
        self.path = self.configs.path_to_store_graphs
        self.colors = ['r', 'g', 'y', 'k', 'm', 'c']
        self.date = self.configs.date_string
        self.color_index = 0

        # Kinda hacky.  This makes it so we can compute and visualize different loss functions
        # I think the way to fix this is to have separate configs for visualization than for training
        self.results_obj.configs.results_loss_function = self.configs.results_loss_function
        self.results_obj.generate_performance_results()

    def plot_all(self):
        self.plot_learning_curve()
        self.plot_actual_vs_predicted_curve()

    def new_color(self):
        col = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return col


    def scatter_data(self, estimators, sizes, x_name, y_name, alpha=1, do_scatter_plot=False):
        legend = []
        for estimator in estimators:
            estimator_results = self.results_obj.estimator_dict[estimator]
            x, y = estimator_results.get_plot_arrays(sizes, (x_name, y_name))
            if do_scatter_plot:
                plt.scatter(x, y, c=self.new_color(), alpha=alpha)
            else:
                plt.plot(x, y, c=self.new_color(), alpha=alpha,lw=5)
            legend.append(estimator + ' ' + y_name)
        return legend


    def plot_learning_curve(self):

        plot_string = 'Learning_Curve'
        plt.figure(self.fig_number)
        estimators = self.results_obj.estimator_names
        training_sizes = self.results_obj.configs.training_sizes
        l1 = self.scatter_data(estimators, training_sizes, 'training_size', 'test_error')
        l2 = self.scatter_data(estimators, training_sizes, 'training_size', 'train_error')


        # add legend and other labels
        self.set_plot_captions(plot_string,
                               'Training Size (Number of Targets)',
                               LossFunction.get_loss_function_display_name(self.configs.results_loss_function),
                               legend_list=l1+l2)

        plt.savefig(self.path + '/' + self.date + '_' + plot_string + '.jpg')

        self.fig_number += 1


    def plot_actual_vs_predicted_curve(self):
        #TODO: This doesn't work anymore.
        plot_string = 'Predicted_vs_Actual_Scatter'
        sizes = self.results_obj.configs.training_sizes
        for estimator_name in self.results_obj.estimator_names:
            plt.figure(self.fig_number)
            for i, size in enumerate(sizes):

                rows_of_subplot = len(sizes)
                plt.subplot(rows_of_subplot*100 + 4*10 + i)
                self.scatter_data([estimator_name], [size], 'test_actual', 'test_predicted',
                                  alpha=1, do_scatter_plot=True)

                x = np.linspace(0.0, 1.0, 1000)
                plt.plot(x, x, color='b',linestyle='--')

                title_string = estimator_name[0:3] + ' and size of ' + str(size)
                self.set_plot_captions(title_string, 'Actual Value', 'Predicted Value')


            plt.savefig(self.path + '/' +self.date + '_' + estimator_name + '_' +'_' + plot_string + '.jpg')
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
