__author__ = 'Evan Racah'
from HelperFunctions import recall
from Learn import Learn
from Visualization import Visualization
import Configs


def recall_main_results(Configs):
    main_results = recall(Configs.path_to_store_results, Configs.save_results_file_name)
    return main_results


def learn_main_results():
    #learner = Learn(path_to_targets, estimators, test_size, path_to_store_results, results_filename)
    learner = Learn(Configs)
    Main_results = learner.run_grid_search()
    Main_results.save_data()
    return Main_results



main_results = learn_main_results()
#main_results = recall_main_results(Configs)
viz = Visualization(main_results, Configs)
viz.plot_all()
viz.show()

















