__author__ = 'Evan Racah'

from matplotlib import pyplot as plt
import math
import Configs

class Visualization(object):

    def __init__(self, main_result_obj,configs):
        self.fig_number = 1
        self.results_obj = main_result_obj
        self.path = configs.path_to_store_graphs
        self.colors = ['r', 'b', 'g', 'y', 'k', 'm', 'c']
        self.date = configs.date_string

    def plot_all(self):
        self.plot_learning_curve()
        self.plot_actual_vs_predicted_curve()

    def plot_learning_curve(self):
        plot_string = 'Learning_Curve'
        #pick an unused figure to plot on
        plt.figure(self.fig_number)

        #for every estimator get the train and test errors vectors and the corresponding
        #training size vectors and plot them both
        for index, estimator_name in enumerate(self.results_obj.estimator_names):
            # print estimator_name
            sizes, test_errors = self.results_obj.estimator_dict[estimator_name].get_test_prediction_error_data()
            sizes, train_errors = self.results_obj.estimator_dict[estimator_name].get_train_prediction_error_data()
            plt.scatter(sizes, test_errors, c=self.colors[2 * index])
            plt.scatter(sizes, train_errors, c=self.colors[2 * index + 1])

        #make list to pass to legend function by taking
        #the union of every "<estimator_name> test" and "<estimator_name> train" strings
        legend_list = [[name + 'test', name + 'train'] for name in self.results_obj.estimator_names]
        leg_list = []
        for sub_list in legend_list:
            leg_list = leg_list + sub_list

        # add legend and other labels
        self.set_plot_captions(plot_string,
                               'Training Size (Number of Targets)',
                               'Mean Squared Error',
                               legend_list=leg_list)
        plt.savefig(self.path + plot_string + '/'+ self.date + '_'+ plot_string + '.jpg')

        self.fig_number += 1


    def plot_actual_vs_predicted_curve(self):
        plot_string = 'Predicted_vs_Actual_Scatter'

        #for every estimator get predictions and answers at every training size
        for estimator_name in self.results_obj.estimator_names:
            #test_actuals and test_predictions are each an array containing multiple arrays each corresponding
            #to each training size from sizes. These subarrays contain the actual
            #gdt_ts value of all the models of the the test set and the predicted
            #values using an estimator trained with number of models
            #in the corresponding sizes array
            sizes, test_actuals, test_predictions = self.results_obj.estimator_dict[
                estimator_name].get_actuals_and_predictions_data()

            #plot a scatter plot for each
            plt.figure(self.fig_number)

            for i, size in enumerate(sizes):

                #get correct subplot
                rows_of_subplot = math.ceil(math.sqrt(len(sizes)))
                plt.subplot(rows_of_subplot*100 + rows_of_subplot*10 + i)
                plt.scatter(test_actuals[i], test_predictions[i], alpha=0.01)


                title_string = estimator_name[0:2] + ' and size of ' + str(size)
                self.set_plot_captions(title_string,'Actual Value', 'Predicted Value')


            plt.savefig(self.path + plot_string + '/' +self.date + '_' + estimator_name + '_' +'_' + plot_string + '.jpg')
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
