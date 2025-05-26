from datetime import date, datetime
import argparse
from populations import FlexiblePopulation, AdaptiveFlexiblePopulation
import numpy as np
from network import sigmoid_activation
from evolution import cmaes_mackey_glass_signal_gen_multi_t_existing_es
from simulator import NetworkSimulator
from config import propagation_vel, get_p_dict_like_p3_MG
import os
import pickle as pkl


if __name__ == '__main__':
    # Runs hyperparameter optimization for DDNs and ESNs in various heterogeneity settings for the Mackey-Glass task,
    # based on existing intermediate results

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", action="store", type=str, default="./", help="Evolution results data path")
    parser.add_argument("-n", "--exponent_range", action="store",
                        help="range of mackey-glass exponent values to be used", nargs=2, type=float,
                        default=[10, 10])
    parser.add_argument("-b", "--bcm", action="store_true", help="Run experiment with adaptive synapses")

    # Filename flags
    parser.add_argument("-s", "--suffix", action="store", help="filename suffix", type=str, default='continue')

    args = parser.parse_args()
    config = vars(args)
    path = config['path']

    # Command line config parameters
    suffix = config['suffix']  # Added at the end of save filename
    n_range = config['exponent_range']
    adaptive = config['bcm']

    if len(suffix) > 0:
        suffix = '_' + suffix
    net_type_name = 'BL'
    max_delay = 0.001

    print("Load past run")
    with open(path, 'rb') as f:
        old_results_dict = pkl.load(f)

    if path[-4:] == '.pkl':
        new_path = path[:-4] + suffix
    else:
        assert path[-2:] == '.p'
        new_path = path[:-2] + suffix
    new_dir, new_filename = os.path.split(new_path)
    print('Experiment will be saved as')
    print(new_path + '.pkl')


    evo_params = {
        'es': old_results_dict['evolutionary strategy'],
        'param_hist': old_results_dict['parameters'],
        'val_hist': old_results_dict['validation performance'],
        'std_hist': old_results_dict['cma stds'],
        'start_net':  old_results_dict['example net'],
        'n_unsupervised': 500,
        'n_supervised': 1000,
        'n_validation': 1000,
        'n_seq_unsupervised': 5,
        'n_seq_supervised': 5,
        'n_seq_validation': 5,
        'error_margin':  old_results_dict['error margin'],
        'tau_list':  old_results_dict['tau list'],
        'n_range': n_range,
        'dir': new_dir,
        'name': new_filename,
        'alphas': old_results_dict['alpha grid'],
        'aggregate': np.mean
    }
    if not adaptive:
        evo_params['n_unsupervised'] = 0
        evo_params['n_supervised'] = 1500
        evo_params['n_seq_unsupervised'] = 0

    cmaes_mackey_glass_signal_gen_multi_t_existing_es(**evo_params)
    #
