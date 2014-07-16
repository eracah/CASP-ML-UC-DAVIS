__author__ = 'Evan Racah'
from Learn import Learn
from Visualization import Visualization
import Configs.Configs as cfg
import os.path
import pickle
import HelperFunctions

def create_main_result_file_name(configs):
    return configs.path_to_store_results + configs.save_results_file_name

def main_results_exist(configs):
    file_name = create_main_result_file_name(configs)
    return os.path.isfile(file_name)

def recall_main_results(configs):
    file_name = create_main_result_file_name(configs)
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def learn_main_results(configs):
    learner = Learn(configs)
    Main_results = learner.run_grid_search()
    Main_results.configs = configs
    Main_results.save_data()
    return Main_results


if __name__ == "__main__":
    batch_configs = cfg.BatchConfigs.create_batch_configs()
    # configs = cfg.Configs()
    for configs in batch_configs.all_configs:
        if configs.we_learn_the_data or not main_results_exist(configs):
            learn_main_results(configs)
            print('Done Training!')
        else:
            print('Results already exist')

    viz_configs = cfg.VisualizationConfigs()
    viz = Visualization(viz_configs)
    for configs in batch_configs.all_configs:
        main_results = recall_main_results(configs)
        print('Done Loading Results!')
        viz.add_to_learning_curve_plot(main_results)

    viz.finish_learning_curve_plot()
    # viz.plot_all()

    if configs.show_plots:
        viz.show()

















