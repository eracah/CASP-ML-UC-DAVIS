__author__ = 'Evan Racah'

from matplotlib import pyplot as plt
import math
import numpy as np
class Visualization(object):

    def __init__(self, main_result_obj, configs):
        self.fig_number = 1
        self.results_obj = main_result_obj
        self.configs = configs
        self.path = configs.path_to_store_graphs
        self.colors = ['r', 'b', 'g', 'y', 'k', 'm', 'c']
        self.date = configs.date_string
        self.color_index = 0

    def plot_all(self):
        self.plot_learning_curve()
        self.plot_actual_vs_predicted_curve()

    def new_color(self):
        col = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return col


    def scatter_data(self, estimators, sizes, x_name, y_name, alpha=1):
            legend = []
            for estimator in estimators:
                x, y = self.results_obj.estimator_dict[estimator].get_plot_arrays(sizes, (x_name, y_name))
                plt.scatter(x, y, c=self.new_color(), alpha=alpha)
                legend.append(estimator + ' ' + y_name)
            return legend


    def plot_learning_curve(self):

        plot_string = 'Learning_Curve'
        plt.figure(self.fig_number)
        estimators = self.results_obj.estimator_names
        l1 = self.scatter_data(estimators, self.configs.training_sizes, 'training_size', 'test_prediction_error')
        l2 = self.scatter_data(estimators, self.configs.training_sizes, 'training_size', 'train_prediction_error')


        # add legend and other labels
        self.set_plot_captions(plot_string,
                               'Training Size (Number of Targets)',
                               'Mean Squared Error',
                               legend_list=l1+l2)

        plt.savefig(self.path + '/' + self.date + '_' + plot_string + '.jpg')

        self.fig_number += 1


    def plot_actual_vs_predicted_curve(self):

        plot_string = 'Predicted_vs_Actual_Scatter'
        sizes = self.configs.training_sizes
        #for every estimator get predictions and answers at every training size
        for estimator_name in self.results_obj.estimator_names:
            plt.figure(self.fig_number)
            for i, size in enumerate(sizes):
                #get correct subplot
                rows_of_subplot = math.ceil(math.sqrt(len(sizes)))
                plt.subplot(rows_of_subplot*100 + rows_of_subplot*10 + i)
                self.scatter_data([estimator_name], [size], 'test_actual_values', 'test_predicted_values',alpha=0.01)
                x = np.linspace(0.0, 1.0, 1000)
                plt.plot(x, x, color=self.new_color())

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
