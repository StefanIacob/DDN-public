import numpy as np
import populations
import evolution
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation
from utils import createNARMA, read_config
from datetime import date
import configparser
import os
import sys

if __name__ == '__main__':
    config_p = configparser.ConfigParser()
    config_p.read(sys.argv[1])
    config_dict = read_config(config_p)
    with open(sys.argv[1], 'r') as f:
        config_s = f.read()
    narma_order = int(config_dict['NARMA data']['narma_order'])
    npseed = config_dict['general']['numpy_seed']

    if not(npseed) is None:
        np.random.seed(int(npseed))

    prefix = "NARMA"+ str(narma_order)

    x_range = config_dict['network']['x_range_start']
    y_range = config_dict['network']['y_range_start']
    B_start = int(config_dict['network']['max_delay'])
    propagation_vel = config_dict['network']['propagation_vel']
    growing_net = config_dict['network']['growing']
    delay = False

    x_lim = x_range
    y_lim = y_range

    if B_start > 1:
        delay = True

    if growing_net:
        delay = True
        prefix += "_growing"
        x_lim = None
        y_lim = None

    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    alphas = config_dict['readout']['regularization_parameters']

    N = int(config_dict['network']['n_neurons'])
    k = int(config_dict['network']['n_clusters'])

    insize = 1
    outsize = N - insize

    n_train = int(config_dict['NARMA data']['n_samples_train'])
    n_val = int(config_dict['NARMA data']['n_samples_validation'])
    coefs = config_dict['NARMA data']['coefs']
    data_train = np.array(createNARMA(n_train, narma_order, coefs)).reshape((2, n_train))
    data_val = np.array(createNARMA(n_val, narma_order, coefs)).reshape((2, n_val))

    max_it = int(config_dict['evolution']['n_generations'])
    pop_size = int(config_dict['evolution']['population_size'])
    std = config_dict['evolution']['std']

    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling_start = np.ones((k + 1,)) * 0.5
    bias_mean_start = np.zeros_like(bias_scaling_start)
    weight_scaling_start = np.ones((k + 1, k + 1)) * .3
    weight_scaling_start[0, 1] = 0.9
    weight_mean_start = np.zeros((k + 1, k + 1))
    decay_start = np.ones((k + 1,)) * .95
    activation = None
    if config_dict['network']['activation'] == 'tanh':
        activation = tanh_activation
    elif config_dict['network']['activation'] == 'sigmoid':
        activation = sigmoid_activation

    propagation_vel = config_dict['network']['propagation_vel']
    dt = np.sqrt(width**2 + height**2) / (B_start * propagation_vel)

    start_mu = np.zeros((k, 2))
    start_mu[:, :] = (0, 0)
    start_var = 0.3
    start_corr = np.ones((k,)) * 0
    start_var = np.ones((k, 2)) * start_var
    inhib_start = np.ones((k,)) * 0
    in_loc = config_dict['network']['in_loc']
    conn_start = np.ones((k + 1, k + 1))

    genome_ranges = config_dict['genome ranges']
    fixed_parameters = config_dict['fixed genome']

    if not delay:
        prefix += "_bl"
    else:
        prefix +="_ddn"

    print('Building starting network...')

    lr_mean = np.ones_like(weight_scaling_start) * 0
    lr_scaling = np.ones_like(lr_mean) * 0
    y0_mean = np.ones_like(bias_mean_start) * 0
    y0_scaling = np.ones_like(bias_mean_start) * 0

    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': start_var,
        'correlation': start_corr,
        'inhibitory': inhib_start,
        'connectivity': conn_start,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling_start,
        'weight_mean': weight_mean_start,
        'bias_scaling': bias_scaling_start,
        'bias_mean': bias_mean_start,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        'x_lim': x_lim,
        'y_lim': y_lim,
        'lr_mean': lr_mean,
        'lr_scaling': lr_scaling,
        'y0_mean': y0_mean,
        'y0_scaling': y0_scaling,
        'propagation_vel': propagation_vel,
        'param_ranges': genome_ranges,
        'fixed_params': fixed_parameters
    }

    start_net = populations.GMMPopulationAdaptive(**net_params)

    # from simulator import NetworkSimulator
    # sim = NetworkSimulator(start_net)
    # sim.visualize(np.random.uniform(size=(1000,)))

    dir1 = config_dict['save file']['directory']
    suffix = config_dict['save file']['dirname_suffix']
    dir2 = prefix + '_results_n' + str(N) + "_k" + str(k) + "_date_" + str(date.today())
    if not(suffix is None):
        dir2 += "_" + suffix
    dir = dir1 + "/" + dir2

    if not os.path.isdir(dir1):
        os.mkdir(dir1)

    if not os.path.isdir(dir):
        os.mkdir(dir)

    config_filename = 'exp_config.txt'
    filename = 'results_data'
    with open(dir + "/" + config_filename, 'w') as f:
        f.write(config_s)
    print("dirname: " + dir)
    print("dt ", dt)
    print("B start max ", B_start)
    print("B start real ", start_net.B)
    evolution.cmaes_alg_gma_pop_timeseries_prediction(start_net, data_train, data_val, config_dict, dir=dir, name=filename)
