import numpy as np
import populations
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation
from config import propagation_vel
from evolution import cmaes_alg_gma_pop_signal_gen_adaptive
from datetime import date
import os
import argparse

def params_start_evolution_adaptive(N, buffersize, var_delays, k):
    insize = 1
    outsize = N - insize
    x_range = (0, .002)
    y_range = (0, .004)
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [(i+1) * (width/(k+1)), (i+1) * (height/(k+1))]

    var = np.ones((k, 2)) * width * 0.1
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k, 1))
    in_loc = (x_range[0] + width * 0.01, y_range[0] + height * 0.01)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+1, k+1)) * .25
    weight_scaling = np.ones_like(connectivity) * 1
    lr_mean = np.ones_like(weight_scaling) * 0.1
    lr_scaling = np.ones_like(lr_mean) * 0.01
    weight_mean = np.zeros_like(weight_scaling)
    bias_mean = np.zeros((k+1, ))

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.ones((k + 1,)) * .5
    y0_mean = np.ones_like(bias_mean) * .8
    y0_scaling = np.ones_like(bias_mean) * .1
    decay_start = np.ones((k + 1,)) * .99
    if buffersize > 2:
        max_dist = np.sqrt(width ** 2 + height ** 2)
        longest_delay_needed = max_dist / propagation_vel
        dt = longest_delay_needed / (buffersize-2)
    else:
        dt = .5

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
        'lr_mean': lr_mean,
        'lr_scaling': lr_scaling,
        'y0_mean': y0_mean,
        'y0_scaling': y0_scaling,
        'x_range': x_range,
        'y_range': y_range,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        'buffersize': buffersize,
        # 'theta_window': None,
        'var_delays': var_delays
    }
    return net_params

def params_start_evolution(N, buffersize, k):
    insize = 1
    outsize = N - insize
    x_range = (0, .002)
    y_range = (0, .004)
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [(i+1) * (width/(k+1)), (i+1) * (height/(k+1))]

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
    if buffersize > 2:
        max_dist = np.sqrt(width ** 2 + height ** 2)
        longest_delay_needed = max_dist / propagation_vel
        dt = longest_delay_needed / (buffersize-2)
    else:
        dt = .5

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
        'x_range': x_range,
        'y_range': y_range,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        'buffersize': buffersize,
    }
    return net_params

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-b", "--bcm", action="store_true", help="Run experiment with adaptive synapses")
    parser.add_argument("-e", "--error-margin", action="store", default=.1,
                        help="Define the blind prediction error margin in variance")

    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    adaptive = config['bcm']
    error_margin = float(config['error_margin'])
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1]

    N = 300
    k = 6

    print('Building starting network...')
    if adaptive:
        B = 25
        net_params = params_start_evolution_adaptive(N, B, delay, k)
        start_net = populations.GMMPopulationAdaptive(**net_params)
    else:
        if delay:
            B = 25
        else:
            B = 1
        net_params = params_start_evolution(N, B, k)
        start_net = populations.GMMPopulation(**net_params)
    gens = 100
    pop_size = 20
    dir = 'results-2023-paper'

    if not os.path.exists(dir):
        os.makedirs(dir)
    tau_range = [17, 17]
    n_range = [10, 10]
    name = str(date.today()) + "_average_fit_" + "_b" +str(B)+ "_n"+str(N)+"_k"+str(k)+ \
           "_multiple_sequences_fixed_params_adaptive" + str(adaptive) + "_delays_" + str(delay) \
           + "_e_" + str(error_margin) + "weight_decay"
    print('Running evolution, saving data as ' + str(name))
    print('Delay: ', delay)
    print('BCM: ', adaptive)

    # define fitness measures
    def weakest_eval_fitness(evals):
        fitness = np.min(evals, axis=-1)
        return fitness

    if adaptive:
        cmaes_alg_gma_pop_signal_gen_adaptive(start_net, 500, 1000, 1000, max_it=gens,
                                              pop_size=pop_size, alphas=alphas, dir=dir, name=name,
                                              n_seq_unsupervised=5,
                                              n_seq_supervised=5,
                                              n_seq_validation=5,
                                              error_margin=error_margin,
                                              tau_range=tau_range,
                                              n_range=n_range,
                                              fitness_function=None)

    else:
        cmaes_alg_gma_pop_signal_gen_adaptive(start_net, 0, 1500, 1000, max_it=gens,
                                              pop_size=pop_size, alphas=alphas, dir=dir, name=name,
                                              n_seq_unsupervised=0,
                                              n_seq_supervised=5,
                                              n_seq_validation=5,
                                              error_margin=error_margin,
                                              tau_range=tau_range,
                                              n_range=n_range,
                                              fitness_function=None)


    # cmaes_alg_gma_pop_signal_gen(start_net, mg_train, mg_validate, max_it=gens,
    #                                       pop_size=pop_size, alphas=alphas, dir=dir, name=name)

    # net_params = params_debug(4)
    # net_params = params_single_cluster(20)
    # horizons = []
    # for i in range(10):
    #     net = populations.GMMPopulationAdaptive(**net_params)
    #     horizons.append(eval_candidate_signal_gen(net, mg_train, mg_validate))
    # print('test')
