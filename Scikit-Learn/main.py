__author__ = 'Evan Racah'
from HelperFunctions import recall
from Learn import Learn
from Visualization import Visualization

from ReadConfigs import path, estimators, \
                        training_sizes, parameter_grids, n_folds, scoring, trials_per_size, test_size
#TODO do not put saving in MainResult either
# results_filename = '7-1-2014_Main_Result.dat'
# results_path = './Results/Data/'
# main_results = recall(results_path, results_filename)

learner = Learn(path, estimators, test_size)
main_results = learner.run_grid_search(training_sizes, parameter_grids, trials_per_size, n_folds, scoring)
main_results.save_data()
viz = Visualization(main_results)
viz.plot_all()
viz.show()

















