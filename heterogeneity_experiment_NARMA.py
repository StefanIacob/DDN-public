from datetime import date, datetime
import argparse
from populations import FlexiblePopulation
import numpy as np
from evolution import cmaes_alg_gma_pop_timeseries_prediction_old
from utils import createNARMA30, createNARMA10
from config import propagation_vel, get_p_dict_heterogeneity_exp, get_p_dict_like_p3_NARMA
from network import tanh_activation

if __name__ == '__main__':
    # Runs hyperparameter optimization for DDNs and ESNs in various heterogeneity settings for the NARMA-10 or 30 tasks

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons",
                        type=int, default=300)
    parser.add_argument("-dd", "--distributed_decay", action="store_true", help="Distributed decay")
    parser.add_argument("-cd", "--cluster_decay", action="store_true", help="Different decay per cluster")
    parser.add_argument("-fd", "--fixed_delays", action="store_true", help="don't optimize delays")
    parser.add_argument("-s", "--suffix", action="store", help="filename suffix", type=str, default='')

    args = parser.parse_args()
    config = vars(args)

    # Command line config parameters
    delay = config['delay']  # False for ESNs, True for DDNs
    N = config['neurons']  # Nr of nodes in reservoir
    K = config['clusters']  # Nr of sub-reservoirs/clusters
    distributed_decay = config['distributed_decay'] # False for fixed decay/leak rate, True for distributed
    # decay/leak rate
    per_cluster_decay = config['cluster_decay']  # False for network wide decay parameters, True for
    # cluster-specific parameters
    fixed_delays = config['fixed_delays']  # Keep location configuration fixed to initial values
    suffix = config['suffix']  # Added at the end of save filename

    if len(suffix) > 0:
        suffix = '_' + suffix
    net_type_name = 'BL'
    max_delay = 0.001
    if delay:
        net_type_name = 'DDN'
        max_delay = 12  # Maximum possible delay if DDNs are used. Change according to task memory requirements

    # DDN spatial parameters/coordinates/ranges
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

    p_dict = get_p_dict_like_p3_NARMA(K, x_range, y_range)  # The parameter configuration that was also used for previous
    # paper. Note that below some of the evolution parameters can be changed to be fixed or evolved, and network wide or
    # per cluster depending on the experiment configurations. Decay configuration used in previous paper was fixed per
    # cluster, i.e., decay mean is evolved, has dimension K, decay scaling is fixed at 0 and not evolved.

    dist_decay_name = 'fixed_decay'
    if distributed_decay:
        # Set decay scaling to be evolved and not 0.
        dist_decay_name = 'dist_decay'
        p_dict['decay_scaling']['val'] = np.array(K * [0.2])
        p_dict['decay_scaling']['evolve'] = True

    per_cluster_name = 'per_cluster'
    if not per_cluster_decay:
        # Reduce dimensions of decay parameters from K (per cluster) to 1 (network wide)
        per_cluster_name = 'net_wide'
        p_dict['decay_mean']['val'] = np.array([0.95])
        p_dict['decay_scaling']['val'] = np.array([p_dict['decay_scaling']['val'][0]])

    if fixed_delays:
        # Let the spatial configurations of the network fixed throughout evolution
        suffix += '_fixed_delays'
        p_dict['mu_x']['evolve'] = False
        p_dict['mu_y']['evolve'] = False
        p_dict['variance_x']['evolve'] = False
        p_dict['variance_y']['evolve'] = False
        p_dict['correlation']['evolve'] = False

    activation_func = tanh_activation
    start_net = FlexiblePopulation(N, x_range, y_range, dt, in_loc, size_in, size_out,
                     p_dict, act_func=activation_func)

    # Task data
    data_train = np.array(createNARMA10(8000)).reshape((2, 8000))
    data_val = np.array(createNARMA10(4000)).reshape((2, 4000))

    gens = 200
    pop_size = 25
    reps_per_cand = 5
    alphas = [10e-7, 10e-5, 10e-3]
    dir='heterogeneity_results'
    filename= (str(date.today()) + '_single_task_exp_' + net_type_name + '_' + dist_decay_name + '_' +
               per_cluster_name + suffix)
    print('Experiment will be saved as')
    print(filename + '.pkl')

    # Run CMA-ES hyperparameter evolution
    cmaes_alg_gma_pop_timeseries_prediction_old(start_net, data_train, data_val, gens, pop_size, reps_per_cand, .3,
                                                alphas, dir=dir, name=filename)
