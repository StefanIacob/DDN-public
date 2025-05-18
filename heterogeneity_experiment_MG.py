from datetime import date, datetime
import argparse
from populations import FlexiblePopulation, AdaptiveFlexiblePopulation
import numpy as np
from network import sigmoid_activation
from evolution import cmaes_multitask_narma, cmaes_mackey_glass_signal_gen_adaptive
from simulator import NetworkSimulator
from config import propagation_vel, get_p_dict_like_p3_MG
import os
import pickle as pkl


if __name__ == '__main__':
    # Runs hyperparameter optimization for DDNs and ESNs in various heterogeneity settings for the Mackey-Glass task

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Network type flags
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-b", "--bcm", action="store_true", help="Run experiment with adaptive synapses")
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-ff", "--feedforward", action="store_true", help="Run experiment with 0 reservoir connectivity")

    # Evolution & Task flags
    parser.add_argument("-e", "--error-margin", action="store", default=.1,
                        help="Define the blind prediction error margin in variance")
    parser.add_argument("-t", "--tau_range", action="store", help="range of mackey-glass tau values to be used",
                        nargs=2, type=float, default=[12, 22])
    parser.add_argument("-n", "--exponent_range", action="store",
                        help="range of mackey-glass exponent values to be used", nargs=2, type=float,
                        default=[10, 10])
    parser.add_argument("-m", "--use_median", action="store_true",
                        help="use median of resamlpes in evolution fitness")

    # Heterogeneity flags
    parser.add_argument("-dd", "--distributed_decay", action="store_true", help="Distributed decay")
    parser.add_argument("-cd", "--cluster_decay", action="store_true", help="Different decay per cluster")
    parser.add_argument("-fd", "--fixed_delays", action="store_true", help="don't optimize delays")

    # Filename flags
    parser.add_argument("-s", "--suffix", action="store", help="filename suffix", type=str, default='')

    args = parser.parse_args()
    config = vars(args)

    # Command line config parameters
    delay = config['delay']  # False for ESNs, True for DDNs
    N = config['neurons']  # Nr of nodes in reservoir
    K = config['clusters']  # Nr of sub-reservoirs/clusters
    distributed_decay = config['distributed_decay']  # False for fixed decay/leak rate, True for distributed
    t_range = config['tau_range']  # range of randomly sampled task time parameters throughout evolution
    n_range = config['exponent_range']  # range of randomly sampled task time parameters throughout evolution
    error_margin = config['error_margin']
    adaptive = config['bcm']
    fixed_delays = config['fixed_delays']
    # decay/leak rate
    per_cluster_decay = config['cluster_decay']  # False for network wide decay parameters, True for
    # cluster-specific parameters
    zero_conn = config['feedforward']
    suffix = config['suffix']  # Added at the end of save filename
    use_median = config['use_median']

    if len(suffix) > 0:
        suffix = '_' + suffix
    net_type_name = 'BL'
    max_delay = 0.001
    if delay:
        net_type_name = 'DDN'
        max_delay = 18  # Maximum possible delay if DDNs are used. Change according to task memory requirements

    # DDN spatial parameters/coordinates/ranges
    x_range = [-1, 1]
    y_range = [-1, 1]
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width ** 2 + height ** 2)
    max_time = max_dist / propagation_vel
    dt = max_time / max_delay
    in_loc = (-1, -1)
    size_in = 1
    size_out = N - size_in
    p_dict = get_p_dict_like_p3_MG(K, x_range, y_range)

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

    if zero_conn:
        # Used to test networks with 0 internal connectivity. These are purely feedforward. In the case of DDNs, these
        # function similar to a delay embedding.
        suffix += '_zero_conn'
        p_dict['connectivity']['evolve'] = False
        p_dict['connectivity']['val'] *= 0

    aggregate = np.mean
    if use_median:
        aggregate = np.median
    activation_func = sigmoid_activation
    start_net = AdaptiveFlexiblePopulation(N, x_range, y_range, dt, in_loc, size_in, size_out,
                                   p_dict, act_func=activation_func)
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1]
    dirname = 'heterogeneity_results_MG'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = (str(date.today()) + '_' + net_type_name + '_' + dist_decay_name + '_' +
                per_cluster_name + suffix)
    print('Experiment will be saved as')
    print(filename + '.pkl')

    evo_params = {
        'start_net': start_net,
        'n_unsupervised': 500,
        'n_supervised': 1000,
        'n_validation': 1000,
        'n_seq_unsupervised': 5,
        'n_seq_supervised': 5,
        'n_seq_validation': 5,
        'error_margin': error_margin,
        'tau_range': t_range,
        'n_range': n_range,
        'max_it': 200,
        'pop_size': 20,
        'eval_reps': 5,
        'dir': dirname,
        'name': filename,
        'alphas': alphas,
        'aggregate': aggregate
    }
    if not adaptive:
        evo_params['n_unsupervised'] = 0
        evo_params['n_supervised'] = 1500
        evo_params['n_seq_unsupervised'] = 0

    cmaes_mackey_glass_signal_gen_adaptive(**evo_params)

