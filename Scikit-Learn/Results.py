__author__ = 'Evan Racah'

from sklearn import metrics
from matplotlib import pyplot as plt


class TrainingSampleResult(object):
    def __init__(self, grid_search_object, test_data, train_data, time_to_fit):
        x_test, y_test = test_data
        x_train, y_train = train_data

        self.best_parameter_values = grid_search_object.best_estimator_.get_params()
        self.error_parameter_values = grid_search_object.grid_scores_
        self.test_predicted_values = grid_search_object.best_estimator_.predict(x_test)
        self.test_actual_values = y_test
        self.train_predicted_values = grid_search_object.best_estimator_.predict(x_train)
        self.train_actual_values = y_train
        self.test_prediction_error = metrics.mean_squared_error(self.test_predicted_values, self.test_actual_values)
        self.train_prediction_error = metrics.mean_squared_error(self.train_predicted_values, self.train_actual_values)
        self.time_to_fit = time_to_fit


class EstimatorResult(object):
    '''Estimator Results Class:
    Contains a dictionary with a TrainingSampleResult object for each Training Size
    Also contains method for plotting a scatter actual vs predicted for every training size
    And a method for plotting parameter value vs. error curve'''

    def __init__(self, estimator_name):
        self.name = estimator_name
        self.trainingSampleDict = {}

    def add_training_results(self, training_size, grid_search_object, test_data, train_data, time_to_fit):
        self.trainingSampleDict[training_size] = TrainingSampleResult(grid_search_object, test_data, train_data,
                                                                      time_to_fit)

    def get_test_pred_error(self, train_sample_result):
        #this function is just to make the map function work
        return train_sample_result.test_prediction_error

    def get_train_pred_error(self, train_sample_result):
        #this function is just to make the map function work
        return train_sample_result.train_prediction_error

    def get_test_actuals(self, train_sample_result):
        return train_sample_result.test_actual_values

    def get_test_predicted(self, train_sample_result):
        return train_sample_result.test_predicted_values

    def get_parameter_errors(self,train_sample_result):
        return train_sample_result.self.error_parameter_values


    def get_test_prediction_error_data(self):
        #returns two vectors: vec of all the training sizes and vec of all prediction errors
        return self.trainingSampleDict.keys(), map(self.get_test_pred_error, self.trainingSampleDict.values())

    def get_train_prediction_error_data(self):
        #returns two vectors: vec of all the training sizes and vec of all prediction errors
        return self.trainingSampleDict.keys(), map(self.get_train_pred_error, self.trainingSampleDict.values())

    def get_actuals_and_predictions_data(self):
        return self.trainingSampleDict.keys(), map(self.get_test_actuals, self.trainingSampleDict.values()), \
               map(self.get_test_predicted, self.trainingSampleDict.values())


class MainResult(object):
    '''Main results class:
    Contains a dictionary, which contains an EstimatorResults object for each estimator
    Also has method for plotting all estimators against each other in a learning curve'''

    def __init__(self, estimator_names):
        self.estimator_dict = {}
        self.estimator_names = estimator_names
        self.fig_number = 1
        #add an estimatorResult object for each estimator to the dictionary
        for name in self.estimator_names:
            self.estimator_dict[name] = EstimatorResult(name)


    def add_estimator_results(self, estimator_name, training_size, grid_search_object, test_data, train_data,
                              time_to_fit):

        self.estimator_dict[estimator_name].add_training_results(training_size,
                                                                 grid_search_object,
                                                                 test_data,
                                                                 train_data,
                                                                 time_to_fit)

    def plot_learning_curve(self):
        plt.figure(self.fig_number)
        for estimator_name in self.estimator_names:
            # print estimator_name
            sizes, test_errors = self.estimator_dict[estimator_name].get_test_prediction_error_data()
            sizes, train_errors = self.estimator_dict[estimator_name].get_train_prediction_error_data()
            plt.plot(sizes, test_errors)
            plt.plot(sizes, train_errors)

        legend_list = [[name + 'test', name + 'train'] for name in self.estimator_names]
        leg_list = []
        for sub_list in legend_list:
            leg_list = leg_list + sub_list


        plt.legend(leg_list)
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Training Size (Number of Targets)')
        plt.title('Learning Curve')
        self.fig_number += 1


    def plot_actual_vs_predicted_curve(self):
        for estimator_name in self.estimator_names:
            sizes, test_actuals, test_predictions = self.estimator_dict[
                estimator_name].get_actuals_and_predictions_data()
            plt.figure(self.fig_number)
            for i, size in enumerate(sizes):
                plt.figure(self.fig_number)
                plt.scatter(test_actuals[i], test_predictions[i])
                plt.ylabel('Predicted Value')
                plt.xlabel('Actual Value')
                title_string = 'Predicted vs. Actual for ' + estimator_name + ' and size of ' + str(size)
                plt.title(title_string)
                self.fig_number += 1




    def plot_parameter_error(self):
        pass







