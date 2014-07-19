__author__ = 'Evan Racah'
from os import system
import sys
try:
    from Learn import Learn
except:
    system('module load python matplotlib python-libs/2.7.5')
    system('module load python matplotlib')
    try:
        from Learn import Learn
    except:
        sys.exit('Rehan recommends getting Scikit-Learn')

from Visualization import Visualization
import Configs.Configs as cfg
import os.path
import pickle
import time


def create_main_result_file_name(configs):
    return configs.path_to_store_results + configs.save_results_file_name

def main_results_exist(configs):
    file_name = create_main_result_file_name(configs)
    return os.path.isfile(file_name)

def recall_main_results(configs):
    file_name = create_main_result_file_name(configs)
    file_name = './Results/Data/7-18-2014/RFR,cv_loss_function=P.dat'
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def learn_main_results(configs):
    learner = Learn(configs)
    main_results = learner.run_grid_search()
    main_results.configs = configs
    main_results.save_data()
    return main_results


if __name__ == "__main__":

    t0 = time.time()
    batch_configs = cfg.BatchConfigs.create_batch_configs()
    #for each estimator's conifgs
    for configs in batch_configs.all_configs:
        if configs.we_learn_the_data or not main_results_exist(configs):
            learn_main_results(configs)
            print('Done Training!')
        else:
            print('Results already exist')
    print "time:", time.time() - t0
    viz_configs = cfg.VisualizationConfigs()
    viz = Visualization(viz_configs)
    for configs in batch_configs.all_configs:
        main_results = recall_main_results(configs)
        print('Done Loading Results!')
        viz.add_to_learning_curve_plot(main_results)

    viz.finish_learning_curve_plot()
    #viz.plot_feature_importance_bar_chart()
    # viz.plot_all()

    if viz_configs.show_plots:
        viz.show()

















