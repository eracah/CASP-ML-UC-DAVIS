__author__ = 'Evan Racah'
from Learn import Learn
from Visualization import Visualization
from Configs import Configs
import pickle


def recall_main_results(configs):
    with open(configs.path_to_store_results + configs.save_results_file_name, 'rb') as f:
        return pickle.load(f)


def learn_main_results(configs):
    #learner = Learn(path_to_targets, estimators, test_size, path_to_store_results, results_filename)
    learner = Learn(configs)
    Main_results = learner.run_grid_search()
    Main_results.configs = configs
    Main_results.save_data()
    return Main_results


if __name__ == "__main__":
    configs = Configs.Configs()
    if configs.we_learn_the_data:
        main_results = learn_main_results(configs)
    else:
        main_results = recall_main_results(configs)

    viz = Visualization(main_results, configs)
    viz.plot_all()

    if configs.show_plots:
        viz.show()

















