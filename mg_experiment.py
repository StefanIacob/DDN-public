import numpy as np
import populations
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation
from config import propagation_vel
from evolution import cmaes_alg_gma_pop_signal_gen_adaptive
from datetime import date
import os
import argparse

def params_start_evolution_adaptive(N, var_delays, k, x_lim=None, y_lim=None, dt=.0005):
    insize = 1
    outsize = N - insize
    x_range = x_lim
    y_range = y_lim
    if x_lim is None:
        x_range = (-1, 1)
        y_range = (-1, 1)
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [x_range[0] + (i+1) * (width/(k+1)), y_range[0] + (i+1) * (height/(k+1))]

    var = np.ones((k, 2)) * width * 0.1
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k, 1))
    in_loc = (x_range[0] + width * 0.01, y_range[0] + height * 0.01)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+1, k+1)) * .9
    weight_scaling = np.ones_like(connectivity) * .5
    lr_mean = np.ones_like(weight_scaling) * 0.01
    lr_scaling = np.ones_like(lr_mean) * 0.01
    weight_mean = np.zeros_like(weight_scaling)
    bias_mean = np.zeros((k+1, ))

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.ones((k + 1,)) * .5
    y0_mean = np.ones_like(bias_mean) * .8
    y0_scaling = np.ones_like(bias_mean) * .1
    decay_start = np.ones((k + 1,)) * .99

    activation = sigmoid_activation
    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': var,
        'correlation': corr,
        'inhibitory': inhib_start,
        'connectivity': connectivity,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling,
        'weight_mean': weight_mean,
        'bias_scaling': bias_scaling,
        'bias_mean': bias_mean,
        'lr_mean': lr_mean,
        'lr_scaling': lr_scaling,
        'y0_mean': y0_mean,
        'y0_scaling': y0_scaling,
        'x_lim': x_lim,
        'y_lim': y_lim,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        # 'theta_window': None
    }
    return net_params

def params_start_evolution(N, dt, k, x_lim=None, y_lim=None):
    insize = 1
    outsize = N - insize
    x_range = x_lim
    y_range = y_lim
    if x_lim is None:
        x_range = (-1, 1)
        y_range = (-1, 1)
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [x_range[0] + (i+1) * (width/(k+1)), y_range[0] + (i+1) * (height/(k+1))]

    var = np.ones((k, 2)) * width * 0.1
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k,))
    in_loc = (x_range[0] + width * 0.01, y_range[0] + height * 0.01)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+1, k+1)) * .25
    weight_scaling = np.ones_like(connectivity) * 1
    weight_mean = np.zeros_like(weight_scaling)
    bias_mean = np.zeros((k+1, ))

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.ones((k + 1,)) * .5
    decay_start = np.ones((k + 1,)) * .99

    # activation = tanh_activation
    activation = sigmoid_activation
    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': var,
        'correlation': corr,
        'inhibitory': inhib_start,
        'connectivity': connectivity,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling,
        'weight_mean': weight_mean,
        'bias_scaling': bias_scaling,
        'bias_mean': bias_mean,
        'x_lim': x_lim,
        'y_lim': y_lim,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt
    }
    return net_params

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-b", "--bcm", action="store_true", help="Run experiment with adaptive synapses")
    parser.add_argument("-g", "--grow", action="store_true", help="Run experiment growing network")
    parser.add_argument("-e", "--error-margin", action="store", default=.1,
                        help="Define the blind prediction error margin in variance")
    parser.add_argument("-t", "--tau_range", action="store", help="range of mackey-glass tau values to be used",
                        nargs=2, type=float, default=[12, 22])
    parser.add_argument("-n", "--exponent_range", action="store",
                        help="range of mackey-glass exponent values to be used", nargs=2, type=float,
                        default=[10, 10])
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-ag", "--aggregate", action="store",
                        help="aggregate function for performance. 0 for mean, 1 for median and 2 for minimum",
                        type=int, default=0)

    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    growing_net = config['grow']
    adaptive = config['bcm']
    error_margin = float(config['error_margin'])
    ag = config['aggregate']

    assert not(growing_net and not delay), "not possible to have a growing network without delay"
    assert ag in [0, 1, 2], "for aggregate function choose 0 for mean, 1 for median and 2 for minimum"
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1]
    aggregate = np.mean
    ag_name = "mean_performance"

    if ag == 1:
        aggregate = np.median
        ag_name = "median_performance"
    if ag == 2:
        aggregate = np.min
        ag_name = "min_performance"

    N = config['neurons']
    k = config['clusters']

    B_start = 10
    dt_delay = np.sqrt(8)/(B_start * propagation_vel)
    x_lim = [-1, 1]
    y_lim = [-1, 1]

    dt_no_delay = (np.sqrt(8) / propagation_vel) * 2

    if growing_net:
        x_lim = y_lim = None

    dt = dt_delay
    if not delay and not adaptive:
        dt = dt_no_delay

    print('Building starting network...')
    if adaptive:
        net_params = params_start_evolution_adaptive (N=N, dt=dt, x_lim=x_lim, y_lim=y_lim, var_delays=delay, k=k)
        start_net = populations.GMMPopulationAdaptive(**net_params)
    else:
        net_params = params_start_evolution(N, dt=dt, k=k, x_lim=x_lim, y_lim=y_lim)
        start_net = populations.GMMPopulation(**net_params)

    gens = 200
    pop_size = 20
    dir_1 = 'ADDN_further_experiments'
    dir_2 = "Results N" + str(N) + " K" + str(k)
    dir = os.path.join(dir_1, dir_2)
    if not os.path.exists(dir):
        os.makedirs(dir)

    tau_range = args.tau_range
    n_range = args.exponent_range

    name = str(date.today()) + "_delay_" + str(delay) + "_bcm_" + str(adaptive) + \
           "_growing_" + str(growing_net) + "_" + ag_name
    print('Running evolution, saving data as ' + str(name))
    print('Delay: ', delay)
    print('BCM: ', adaptive)

    if adaptive:
        cmaes_alg_gma_pop_signal_gen_adaptive(start_net, 500, 1000, 1000, max_it=gens,
                                              pop_size=pop_size, alphas=alphas, dir=dir, name=name,
                                              n_seq_unsupervised=5,
                                              n_seq_supervised=5,
                                              n_seq_validation=5,
                                              error_margin=error_margin,
                                              tau_range=tau_range,
                                              n_range=n_range,
                                              fitness_function=None,
                                              aggregate=aggregate)

    else:
        cmaes_alg_gma_pop_signal_gen_adaptive(start_net, 0, 1500, 1000, max_it=gens,
                                              pop_size=pop_size, alphas=alphas, dir=dir, name=name,
                                              n_seq_unsupervised=0,
                                              n_seq_supervised=5,
                                              n_seq_validation=5,
                                              error_margin=error_margin,
                                              tau_range=tau_range,
                                              n_range=n_range,
                                              fitness_function=None,
                                              aggregate=aggregate)