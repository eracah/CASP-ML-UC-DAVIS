__author__ = 'Evan Racah'
from HelperFunctions import recall
from Learn import Learn
from Visualization import Visualization


from Configs import path_to_targets, path_to_store_results, path_to_store_graphs, estimators, \
    training_sizes, parameter_grids, n_folds, scoring, trials_per_size, test_size, n_cores, save_results_file_name,\
    recall_results_file_name


def recall_main_results(results_filename):
    main_results = recall(path_to_store_results, results_filename)
    return main_results


def learn_main_results(results_filename):
    learner = Learn(path_to_targets, estimators, test_size, path_to_store_results, results_filename)
    main_results = learner.run_grid_search(training_sizes, parameter_grids, trials_per_size, n_folds, scoring, n_cores)
    main_results.save_data()
    return main_results



#main_results = learn_main_results(save_results_file_name)
main_results = recall_main_results(recall_results_file_name)
viz = Visualization(main_results, path_to_store_graphs)
viz.plot_all()
# viz.show()

















