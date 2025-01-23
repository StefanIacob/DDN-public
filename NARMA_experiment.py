import numpy as np
import populations
import evolution
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation
from utils import createNARMA, createNARMA30
from datetime import date
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-o", "--order", action="store", help="NARMA order", type=int, default=10)
    parser.add_argument("-ag", "--aggregate", action="store",
                        help="aggregate function for performance. 0 for mean, 1 for median and 2 for minimum",
                        type=int, default=0)

    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    ag = config['aggregate']
    narma_order = config['order']

    prefix = "NARMA" + str(narma_order) + "_old"

    assert ag in [0, 1, 2], "for aggregate function choose 0 for mean, 1 for median and 2 for minimum"
    alphas = [10e-7, 10e-5, 10e-3]

    if ag == 1:
        aggregate = np.median
        ag_name = "median_performance"
        prefix += "_med_perf"
    if ag == 2:
        aggregate = np.min
        ag_name = "min_performance"
        prefix += "_min_perf"

    N = config['neurons']
    k = config['clusters']

    insize = 1
    outsize = N - insize
    if narma_order == 30:
        data_train = np.array(createNARMA30(8000)).reshape((2, 8000))
        data_val = np.array(createNARMA30(4000)).reshape((2, 4000))
        data_test = np.array(createNARMA30(4000)).reshape((2, 4000))
    else:
        data_train = np.array(createNARMA(8000, narma_order)).reshape((2, 8000))
        data_val = np.array(createNARMA(4000, narma_order)).reshape((2, 4000))
        data_test = np.array(createNARMA(4000, narma_order)).reshape((2, 4000))

    max_it = 200
    pop_size = 25
    std = 0.1

    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling_start = np.ones((k + 1,)) * 0.5
    weight_scaling_start = np.ones((k + 1, k + 1)) * .6
    weight_scaling_start[0, 1] = 0.9
    decay_start = np.ones((k + 1,)) * .95
    activation = tanh_activation
    dt_delay = .000008
    dt_bl = .5
    x_lim = [0, .002]
    y_lim = [0, .004]

    start_mu = np.zeros((k, 2))
    width = x_lim[1] - x_lim[0]
    height = y_lim[1] - y_lim[0]
    start_mu[:, :] = (np.mean(x_lim) + 0.2 * width, np.mean(y_lim) + 0.4 * width)
    start_var = (width) * 0.15
    start_corr = np.ones((k,)) * 0.15
    start_var = np.random.uniform(start_var, start_var, (k, 2))
    inhib_start = np.ones((k,)) * 0
    in_loc = start_mu[0]
    conn_start = np.ones((k + 1, k + 1)) * .9

    if not delay:
        dt = dt_bl
        prefix += "_bl"
    else:
        prefix += "_ddn"
        dt = dt_delay

    print('Building starting network...')

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
        'bias_scaling': bias_scaling_start,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        'x_range': x_lim,
        'y_range': y_lim
    }
    start_net = populations.GMMPopulationOld(**net_params)

    dir = 'NARMA-' + str(narma_order) + '_results'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    filename = prefix + '_results_n' + str(N) + "_k" + str(k) + "_date_" + str(date.today())
    print("filename: " + filename)
    print("dt ", dt)
    print("B start real ", start_net.B)

    evolution.cmaes_alg_gma_pop_timeseries_prediction(start_net, data_train, data_val, max_it, pop_size, std=std,
                                                      dir=dir, name=filename, alphas=alphas)
