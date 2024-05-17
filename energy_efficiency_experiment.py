import numpy as np
import populations
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation, get_multi_activation, relu
from config import propagation_vel
from simulator import NetworkSimulator
from reservoirpy import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import os
from datetime import date
import argparse
from evolution import cmaes_alg_gma_pop_signal_gen_adaptive


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
    plt.show()
    plt.axis("tight")

def params_start_evolution_adaptive(N, dt, var_delays, k, net_size_factor=1):
    insize = 1
    outsize = N - insize
    x_lim = None
    y_lim = None
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = .002 * net_size_factor
    height = .003 * net_size_factor

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [(i+1) * (width/(k+1)), (i+1) * (height/(k+1))]

    var = np.ones((k, 2)) * width * 0.1
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k, 1))
    in_loc = (0, 0)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+1, k+1)) * .02
    weight_scaling = np.ones_like(connectivity) * 1.2
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
        'dt': dt,
        'buffersize': None,
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
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-a", "--activation_cost", action="store", help="Energy cost scaler for neuron activation", default=.005)
    parser.add_argument("-s", "--synapse_cost", action="store", help="Energy cost scaler for synaptic transmissions", default=.001)
    parser.add_argument("-p", "--propagation_cost", action="store", help="Energy cost scaler for axon propagation", default=.005)
    parser.add_argument("-t", "--tau_range", action="store", help="range of mackey-glass tau values to be used", nargs=2, type=float, default=[12, 22])
    parser.add_argument("-n", "--exponent_range", action="store", help="range of mackey-glass exponent values to be used", nargs=2, type=float,
                        default=[10, 10])

    np.random.seed(32)
    args = parser.parse_args()
    config = vars(args)
    activation_cost = float(config['activation_cost'])
    synapse_cost = float(config['synapse_cost'])
    propagation_cost = float(config['propagation_cost'])
    tau_range = config['tau_range']
    n_range = config["exponent_range"]
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1]
    mackey_glass = datasets.mackey_glass(20000)

    # mg_train = mackey_glass[:10000]
    # mg_train = np.random.uniform(size=(10000,))
    # mg_validate = mackey_glass[10000:]
    N = 200
    k = 6
    dt = .0005
    net_scaling = 25
    print('Building starting network...')
    net_params = params_start_evolution_adaptive(N, dt, True, k, net_size_factor=net_scaling)
    start_net = populations.GMMPopulationAdaptive(**net_params)

    sim = NetworkSimulator(start_net)
    sim.visualize(mackey_glass)

    # import utils
    # utils.eval_candidate_signal_gen_multiple_random_sequences_adaptive_budget(start_net, 5, 5, 5, 500, 1000, 600,
    #                                                                                                      alphas=alphas,
    #                                                                                                      error_margin=error_margin,
    #                                                                                                      tau_range=[17, 17],
    #                                                                                                      n_range=[10, 10])

    gens = 100
    pop_size = 20
    dir = 'adaptive_efficient_cma_es_results'

    if not os.path.exists(dir):
        os.makedirs(dir)

    name = str(date.today()) + "energy_efficiency_random_tau_" + "propagation_cost_" + str(propagation_cost)
    cmaes_alg_gma_pop_signal_gen_adaptive(start_net, 500, 1000, 1000, max_it=gens,
                                          pop_size=pop_size, alphas=alphas, dir=dir, name=name,
                                          n_seq_unsupervised=5,
                                          n_seq_supervised=5,
                                          n_seq_validation=5,
                                          tau_range=tau_range,
                                          n_range=n_range,
                                          fitness_function=None,
                                          activation_cost=activation_cost,
                                          synapse_cost=synapse_cost,
                                          propagation_cost=propagation_cost)
