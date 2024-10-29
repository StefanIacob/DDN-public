from datetime import date, datetime
import argparse
from populations import FlexiblePopulation
import numpy as np
from evolution import cmaes_multitask_narma
from utils import createNARMA10, createNARMA30, inputs2NARMA
from config import propagation_vel, get_p_dict_heterogeneity_exp
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-w", "--weighted", action="store_true", help="Use weighted multi-task score")
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-dd", "--distributed", action="store_true", help="Distributed location parameter")
    parser.add_argument("-cd", "--cluster_diff", action="store_true", help="Different locations per cluster")
    parser.add_argument("-mt", "--multitask", action="store_true", help="Optimize for multiple tasks")
    parser.add_argument("-s", "--suffix", action="store", help="filename suffix", type=str, default='')
    parser.add_argument("-sd", "--seed", action="store", help="random seed", type=int, default=3)


    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    N = config['neurons']
    K = config['clusters']
    weighted = config['weighted']
    multi_task = config['multitask']
    distributed = config['distributed']
    cluster_diff = config['cluster_diff']
    suffix = config['suffix']
    seed = config['seed']
    np.random.seed(seed)
    if len(suffix) > 0:
        suffix = '_' + suffix

    net_type_name = 'BL'
    max_delay = 0.1
    if delay:
        net_type_name = 'DDN'
        max_delay = 15
    x_range = [-.01, .01]
    y_range = [-.01, .01]
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width**2 + height**2)
    max_time = max_dist/propagation_vel
    dt = max_time/max_delay
    in_loc = (-1, 1)
    size_in = 1
    size_out = N - size_in
    start_location_var = 0.002
    start_locatation_mean_var = 0.008
    start_weight_mean = .5
    start_weight_var = .5
    start_bias_mean = 0
    start_bias_var = .5
    p_dict = get_p_dict_heterogeneity_exp(K, x_range, y_range, start_location_var, start_locatation_mean_var,
                                          start_weight_mean, start_weight_var, start_bias_mean, start_bias_var)
    dist_name = 'dist'
    if not distributed:
        dist_name = 'fixed'
        p_dict['bias_scaling']['val'] = np.array([0])
        p_dict['weight_scaling']['val'] = np.array([0])
        p_dict['bias_scaling']['evolve'] = False
        p_dict['weight_scaling']['evolve'] = False

    per_cluster_name = 'per_cluster'
    if not cluster_diff:
        per_cluster_name = 'net_wide'
        p_dict['bias_mean']['val'] = np.array([start_bias_mean])
        p_dict['bias_scaling']['val'] = np.array([start_bias_var])
        p_dict['weight_mean']['val'] = np.array([start_weight_mean])
        p_dict['weight_scaling']['val'] = np.array([start_weight_var])

    start_net = FlexiblePopulation(N, x_range, y_range, dt, in_loc, size_in, size_out,
                     p_dict)

    inputs_train, labels_train_10 = createNARMA10(8000)
    inputs_val, labels_val_10 = createNARMA10(4000)

    labels_train_30 = inputs2NARMA(inputs_train, system_order=30, coef=[.2, .04, 1.5, .001])
    labels_val_30 = inputs2NARMA(inputs_val, system_order=30, coef=[.2, .04, 1.5, .001])

    data = {
        'train': {
            'inputs': inputs_train,
            'labels': {
                'NARMA-10': labels_train_10,
                'NARMA-30': labels_train_30,
            }
        },
        'validation': {
            'inputs': inputs_val,
            'labels': {
                'NARMA-10': labels_val_10,
                'NARMA-30': labels_val_30,
            }
        }
    }

    def weighting_func(task_list, task_axis=-1):
        task_list = task_list**2
        n_tasks = task_list.shape[task_axis]
        pop_size = task_list.shape[task_axis + 1]
        weights = [5/6, 1/6]
        pop_score = np.zeros((pop_size,))
        for t in range(n_tasks):
            task = np.take(task_list, indices=t, axis=task_axis)
            pop_score += task * weights[t]
        return pop_score

    gens = 200
    pop_size = 20
    reps_per_cand = 5
    alphas = [10e-7, 10e-5, 10e-3]
    dir='weight_heterogeneity_results'

    ### Visualization Code ###
    from simulator import NetworkSimulator
    sim = NetworkSimulator(start_net)
    sim.visualize(data['train']['inputs'])
    ### ------------------ ###

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Folder created: {dir}")

    filename= str(date.today()) + '_multi_task_exp_weighed_' + net_type_name + '_' + dist_name + '_' + per_cluster_name + suffix
    print('Experiment will be saved as')
    print(filename + '.pkl')
    if weighted:
        cmaes_multitask_narma(start_net, data, gens, pop_size, reps_per_cand, .3,
                                                    alphas, dir=dir, name=filename, weighing_func=weighting_func)
    else:
        cmaes_multitask_narma(start_net, data, gens, pop_size, reps_per_cand, .3,
                              alphas, dir=dir, name=filename)