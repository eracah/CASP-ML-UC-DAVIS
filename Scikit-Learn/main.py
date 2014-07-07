__author__ = 'Evan Racah'
from Learn import Learn
from Visualization import Visualization
import configs
import pickle


def recall_main_results(Configs):
    with open(Configs.path_to_store_results + Configs.save_results_file_name, 'rb') as f:
        return pickle.load(f)


def learn_main_results():
    #learner = Learn(path_to_targets, estimators, test_size, path_to_store_results, results_filename)
    learner = Learn(configs)
    Main_results = learner.run_grid_search()
    Main_results.save_data()
    return Main_results


if __name__ == "__main__":

    if configs.we_learn_the_data:
        main_results = learn_main_results()
    else:
        main_results = recall_main_results(configs)

    viz = Visualization(main_results, configs)
    viz.plot_all()

    if configs.show_plots:
        viz.show()

















