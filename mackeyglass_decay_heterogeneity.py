from datetime import date, datetime
import argparse
from populations import FlexiblePopulation
import numpy as np
from evolution import cmaes_multitask_narma, cmaes_alg_gma_pop_signal_gen_adaptive
from simulator import NetworkSimulator
from config import propagation_vel, get_p_dict_heterogeneity_exp
import os
import pickle as pkl


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-dd", "--distributed", action="store_true", help="Distributed location parameter")
    parser.add_argument("-cd", "--cluster_diff", action="store_true", help="Different locations per cluster")
    parser.add_argument("-s", "--suffix", action="store", help="filename suffix", type=str, default='')
    parser.add_argument("-sd", "--seed", action="store", help="random seed", type=int, default=3)

    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    N = config['neurons']
    K = config['clusters']
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
    max_dist = np.sqrt(width ** 2 + height ** 2)
    max_time = max_dist / propagation_vel
    dt = max_time / max_delay
    in_loc = (-1, 1)
    out_loc = (1, -1)
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
        p_dict['variance_x']['val'] = np.array([0] * K)
        p_dict['variance_y']['val'] = np.array([0] * K)
        p_dict['variance_x']['evolve'] = False
        p_dict['variance_y']['evolve'] = False

    per_cluster_name = 'per_cluster'
    if not cluster_diff:
        per_cluster_name = 'net_wide'
        p_dict['mu_x']['val'] = np.array([0])
        p_dict['mu_y']['val'] = np.array([0])
        var_x = p_dict['variance_x']['val'][0]
        var_y = p_dict['variance_x']['val'][0]
        p_dict['variance_x']['val'] = np.array([var_x])
        p_dict['variance_y']['val'] = np.array([var_y])

    start_net = FlexiblePopulation(N, x_range, y_range, dt, in_loc, size_in, size_out,
                                   p_dict)

    # ### Visualization Code ###
    # from simulator import NetworkSimulator
    # from reservoirpy.datasets import mackey_glass
    # mg_example = mackey_glass(1000)
    # sim = NetworkSimulator(start_net)
    # net_data = sim.get_network_data(mg_example)
    # ### ------------------ ###

    n_unsupervised = 0
    n_supervised = 1000
    n_validation = 1000
    max_it = 200
    pop_size = 20
    eval_reps = 5
    dir_name = 'MG_heterogeneity_results'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Folder created: {dir_name}")

    filename = str(
        date.today()) + 'mackey_glass_heterogeneity_' + net_type_name + '_' + dist_name + '_' + per_cluster_name + suffix
    print('Experiment will be saved as')
    print(filename + '.pkl')

    alphas = [10e-7, 10e-5, 10e-3]
    cmaes_alg_gma_pop_signal_gen_adaptive(start_net, n_unsupervised=n_unsupervised, n_supervised=n_supervised,
                                              n_validation=n_validation, max_it=max_it, pop_size=pop_size,
                                              eval_reps=eval_reps, dir=dir_name, name=filename, alphas=alphas,
                                              n_seq_unsupervised=0, n_seq_supervised=5, n_seq_validation=5,
                                              error_margin=.1,
                                              tau_range=[12, 22], n_range=[10, 10])
    # sim = NetworkSimulator(start_net)
    # sim.visualize(inputs_train)

