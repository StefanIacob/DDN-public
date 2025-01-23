import numpy as np
import populations
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation, get_multi_activation, relu
from config import propagation_vel
from simulator import NetworkSimulator
from reservoirpy import datasets
from evolution import cmaes_alg_gma_pop_signal_gen, cmaes_alg_gma_pop_signal_gen_adaptive, continue_cmaes_adaptive
from datetime import date
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import LogNorm
from sklearn import mixture
import configparser
from utils import read_config


def GMM_plot(net):
    n_samples = 300

    # generate random sample, two components
    np.random.seed(0)
    k = net.k
    mu = np.stack([net.mu_x, net.mu_y], 0)
    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=k, covariance_type="full", weights_init=net.mix,
                                  means_init=mu)
    clf.means_ = mu
    clf.covariances_ = net.covariances
    clf.init_params()
    clf.predict([[0, 0]])

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    # plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.show()

def params_start_evolution_adaptive(N, k, x_lim=None, y_lim=None, dt=.0005):
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

def params_four_layer(N, buffersize):
    # N = 301
    insize = 1
    outsize = N - insize
    x_range = (0, .002)
    y_range = (0, .004)
    k = 4
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [(i+1) * (width/(k+1)), (i+1) * (height/(k+1))]

    var = np.ones((k, 2)) * width * 0.1
    corr = np.ones((k,)) * -0.9
    var[2, :] = [width * .05, width * .12]
    corr[1] = .3
    inhib_start = np.array([.0, .0, .0, .0])
    in_loc = (x_range[0] + width * 0.01, y_range[0] + height * 0.01)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectvitiy = np.array(
        [
            [0.8, 0, .5, 0, 1],
            [.9, 0.8, .3, 0, 1],
            [0, .9, 0.8, 0, 1],
            [0, 0.2, .9, 0.8, 1],
            [0, 0, 0, 0, 0]
        ]
    )
    weight_scaling = np.array(
        [
            [.5, 0, 0, 0, 3],
            [.7, .6, 0, .7, 3],
            [0, .7, .7, 0, 3],
            [0, 0, .8, .5, 3],
            [0, 0, 0, 0, 3]
        ]
    )

    lr_mean = np.ones_like(weight_scaling) * 0.1
    lr_mean[0, -1] = 0.01
    lr_scaling = np.ones_like(lr_mean) * 0.01
    weight_mean = np.zeros_like(weight_scaling)
    bias_mean = np.zeros((k+1, ))

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.ones((k + 1,)) * .5
    y0_mean = np.ones_like(bias_mean) * .8
    y0_scaling = np.ones_like(bias_mean) * .1
    decay_start = np.ones((k + 1,)) * .9
    if buffersize > 2:
        max_dist = np.sqrt(width ** 2 + height ** 2)
        longest_delay_needed = max_dist / propagation_vel
        dt = longest_delay_needed / (buffersize-2)
    else:
        dt = .5

    activation = tanh_activation
    # activation = sigmoid_activation
    LONG_BUFF = 0
    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': var,
        'correlation': corr,
        'inhibitory': inhib_start,
        'connectivity': connectvitiy,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling,
        'weight_mean': weight_mean,
        'bias_scaling': bias_scaling,
        'bias_mean': bias_mean,
        # 'lr_mean': lr_mean,
        # 'lr_scaling': lr_scaling,
        # 'y0_mean': y0_mean,
        # 'y0_scaling': y0_scaling,
        'x_range': x_range,
        'y_range': y_range,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        'buffersize': buffersize + LONG_BUFF,
        # 'theta_window': None
    }
    return net_params


def get_mg_labels(mg_series, nr_of_steps):
    labels = []
    for i in range(len(mg_series) - nr_of_steps):
        labels.append(np.reshape(mg_series[i:i+nr_of_steps], (nr_of_steps,)))
    labels = np.stack(labels, axis=0)
    return labels


if __name__ == '__main__':

    # args = parser.parse_args()
    config_p = configparser.ConfigParser()
    config_p.read(sys.argv[1])
    config_dict = read_config(config_p)
    with open(sys.argv[1], 'r') as f:
        config_s = f.read()

    r_seed = int(config_dict['general']['numpy_seed'])
    np.random.seed(r_seed)
    error_margin = float(config_dict['evolution']['error_margin'])
    # ag = config['aggregate']
    x_range = config_dict['network']['x_range_start']
    y_range = config_dict['network']['y_range_start']
    B_start = int(config_dict['network']['max_delay'])
    propagation_vel = config_dict['network']['propagation_vel']
    growing_net = config_dict['network']['growing']
    delay = False

    prefix = "MG"

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
    dt = np.sqrt(width ** 2 + height ** 2) / (B_start * propagation_vel)

    N = int(config_dict['network']['n_neurons'])
    k = int(config_dict['network']['n_clusters'])
    start_mu = np.zeros((k, 2))
    start_mu[:, :] = (0, 0)
    start_var = 0.3
    start_corr = np.ones((k,)) * 0
    start_var = np.ones((k, 2)) * start_var
    start_inhib = np.ones((k,)) * 0
    in_loc = config_dict['network']['in_loc']
    start_conn = np.ones((k + 1, k + 1))

    genome_ranges = config_dict['genome ranges']
    fixed_parameters = config_dict['fixed genome']
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

    lr_mean = np.ones_like(weight_scaling_start) * 0
    lr_scaling = np.ones_like(lr_mean) * 0
    y0_mean = np.ones_like(bias_mean_start) * 0
    y0_scaling = np.ones_like(bias_mean_start) * 0

    insize = 1
    outsize = N - insize

    print('Building starting network...')
    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': start_var,
        'correlation': start_corr,
        'inhibitory': start_inhib,
        'connectivity': start_conn,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling_start,
        'weight_mean': weight_mean_start,
        'bias_scaling': bias_scaling_start,
        'bias_mean': bias_mean_start,
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
        'propagation_vel': propagation_vel,
        'param_ranges': genome_ranges,
        'fixed_params': fixed_parameters
    }

    gens = 200
    pop_size = 20
    dir_1 = 'ADDN_further_experiments'
    dir_2 = "Results N" + str(N) + " K" + str(k) + "full_connectivity"
    dir = os.path.join(dir_1, dir_2)
    # if not os.path.exists(dir_1):
    #     os.makedirs(dir_1)
    if not os.path.exists(dir):
        os.makedirs(dir)

    tau_range = [17, 17]
    n_range = [10, 10]

    name = str(date.today()) + "_delay_" + str(delay) + "_bcm_" + str(adaptive) + \
           "_growing_" + str(growing_net) + "_" + ag_name
    print('Running evolution, saving data as ' + str(name))
    print('Delay: ', delay)
    print('BCM: ', adaptive)

    # define fitness measures
    def weakest_eval_fitness(evals):
        fitness = np.min(evals, axis=-1)
        return fitness
    #
    # EXISTING_FILE = ('adaptive_cma_es_results',
    #                  '2023-08-18_average_fit__b25_n300_k6_multiple_sequences_fixed_params_adaptiveTrue_delays_True_e_0.1weight_decay')
    EXISTING_FILE = None

    if not (EXISTING_FILE is None):
        continue_cmaes_adaptive(EXISTING_FILE[0], EXISTING_FILE[1])
    else:
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


    # cmaes_alg_gma_pop_signal_gen(start_net, mg_train, mg_validate, max_it=gens,
    #                                       pop_size=pop_size, alphas=alphas, dir=dir, name=name)
    #
    # net_params = params_debug(4)
    # net_params = params_single_cluster(20)
    # horizons = []
    # for i in range(10):
    #     net = populations.GMMPopulationAdaptive(**net_params)
    #     horizons.append(eval_candidate_signal_gen(net, mg_train, mg_validate))
    # print('test')
