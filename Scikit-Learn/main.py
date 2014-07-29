__author__ = 'Evan Racah'
from os import system
try:
    from Learn import Learn
except ImportError:
    print 'ImportError'
    system('module load python matplotlib python-libs/2.7.5')
    from Learn import Learn

import Configs.Configs as Configs

general_configs = Configs.Configs()
if general_configs.generate_plots:
    from Visualization import Visualization

import os.path
import pickle
import time
# try:
#     from mpi4py import MPI
# except ImportError:
#     system('module load mpi4py')
#     from mpi4py import MPI


class Main(object):
    def __init__(self, cfg, rank, size):
        # batch_configs = cfg.BatchConfigs.create_batch_configs_for_parallelization([1, 2, 4])
        #for each estimator's conifgs
        self.num_proc = size
        self.processor_id = rank
        self.configs_module = cfg
        self.batch_configs = cfg.BatchConfigs.create_batch_configs()

    def create_main_result_file_name(self,configs):
        return configs.path_to_store_results + configs.save_results_file_name

    def main_results_exist(self, configs):
        file_name = self.create_main_result_file_name(configs)
        return os.path.isfile(file_name)


    def recall_main_results(self, configs):
        file_name = self.create_main_result_file_name(configs)
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def learn_main_results(self,configs):
        learner = Learn(configs)
        main_results = learner.run_grid_search()
        main_results.save_data()
        return main_results

    def generate_results(self):
        for index, configs in enumerate(self.batch_configs.all_configs):
            if self.processor_id == (index % self.num_proc):
                if configs.we_learn_the_data or not self.main_results_exist(configs):
                    self.learn_main_results(configs)
                    print('Done Training!')
                else:
                    print('Results already exist')

        if configs.generate_plots:
            self.visualize()

    def visualize(self):
        viz_configs = self.configs_module.VisualizationConfigs()
        viz = Visualization(viz_configs)
        for configs in self.batch_configs.all_configs:
            main_results = self.recall_main_results(configs)
            print('Done Loading Results!')
            viz.add_to_learning_curve_plot(main_results)

        viz.finish_learning_curve_plot()
        #viz.plot_feature_importance_bar_chart()
        # viz.plot_all()

        if viz_configs.show_plots:
            viz.show()


def run_main(args):
    job_id = args[0]
    num_jobs = args[1]

    t0 = time.time()
    main = Main(Configs, job_id, num_jobs)
    main.generate_results()

    if job_id == 0:
        print "time:", time.time() - t0

    general_configs = Configs.Configs()
    if general_configs.generate_plots:
        main.visualize()

if __name__ == "__main__":
    run_main([0, 1])




















