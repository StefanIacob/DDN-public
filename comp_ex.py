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
from matplotlib.colors import LogNorm
from sklearn import mixture
import argparse
import pickle as pkl
import utils

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
        # 'theta_window': None,
        'var_delays': var_delays
    }
    return net_params

def get_comp_net(N, k, x_range, y_range, c_inter, c_intra, s_inter, s_intra, dt):
    insize = 1
    outsize = N - insize
    if x_range is None:
        x_range = (-1, 1)
        y_range = (-1, 1)
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    # location
    start_mu = np.random.uniform(x_range[0], x_range[1], (k, 2))
    # for i in range(k):
    #     start_mu[i, :] = [x_range[0] + (i + 1) * (width / (k + 1)), y_range[0] + (i + 1) * (height / (k + 1))]

    var = np.ones((k, 2)) * width * 0.05
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k,))
    in_loc = (x_range[0] + width * 0.01, y_range[0] + height * 0.01)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k + 1, k + 1)) * c_inter
    np.fill_diagonal(connectivity, c_intra)
    connectivity[:, -1] = 1
    connectivity[-1, :] = 0
    weight_scaling = np.ones_like(connectivity) * s_inter
    np.fill_diagonal(weight_scaling, s_intra)
    weight_scaling[:, -1] = .6
    weight_scaling[-1, :] = 0

    weight_mean = np.zeros_like(weight_scaling)
    bias_mean = np.zeros((k + 1,))

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

    var = np.ones((k, 2)) * width * 8
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

def eval_net(network, train, val, alphas):
    _, perf, _ = utils.eval_candidate_lag_gridsearch_NARMA(network, train, val, warmup=400, lag_grid=range(0, 10), alphas=alphas)
    return np.min(perf)

def get_coco(net_act, W, comp_div, variance_limit, D=None):
    """
    net_act: N by T float where N is the number of nodes and T is the amount of simulation steps
    W: N by N float where N is the number of nodes
    comp_div: compartment division or cluster division. Int array of size N specifying to which cluster each node belongs
    variance_limit: explained variance threshold for determining dimensionality
    D: N by N Delay matrix
    """

    comp_div = comp_div[1:]
    cluster_acts = []
    connection_acts = []
    k = np.max(comp_div) + 1
    for i in range(k):
        cluster_act = net_act[np.argwhere(comp_div==i), :]
        cluster_act = np.reshape(cluster_act, (cluster_act.shape[0], cluster_act.shape[-1]))
        cluster_acts.append(cluster_act)

        rest_act = net_act[np.argwhere(comp_div != i), :]
        rest_act = np.reshape(rest_act, (rest_act.shape[0], rest_act.shape[-1]))
        rep_rest_act = np.repeat(rest_act, cluster_act.shape[0], axis=0)
        W_net_cluster = W[np.argwhere(comp_div!=i), :]
        W_net_cluster = W_net_cluster[:, 0, np.argwhere(comp_div == i)]
        W_net_cluster_flat = np.reshape(W_net_cluster, (W_net_cluster.shape[0] * W_net_cluster.shape[1],1))
        W_net_cluster_flat = np.repeat(W_net_cluster_flat, rep_rest_act.shape[-1], 1)
        connection_act = rep_rest_act * W_net_cluster_flat
        if not(D is None):
            D_net_cluster = D[np.argwhere(comp_div!=i), :]
            D_net_cluster = D_net_cluster[:, 0, np.argwhere(comp_div == i)]
            D_net_cluster_flat = np.reshape(D_net_cluster, (D_net_cluster.shape[0] * D_net_cluster.shape[1],))
            D_max = np.max(D_net_cluster_flat)
            connection_act_delayed = [connection_act[i, D_net_cluster_flat[i]:-D_max-1+D_net_cluster_flat[i]] for i in range(len(D_net_cluster_flat))]
            connection_act_delayed_pruned = []
            while len(connection_act_delayed) > 0:
                single_connection = connection_act_delayed.pop()
                if np.any(single_connection != 0):
                    connection_act_delayed_pruned.append(single_connection)

            connection_acts.append(connection_act_delayed_pruned)
        else:
            connection_acts.append(connection_act)

    cluster_dim = []
    for cluster_act in cluster_acts:
        _, s, _ = np.linalg.svd(cluster_act)
        dim = 0
        var_explained = 0
        while var_explained < variance_limit:
            var_explained = np.sum(s[:dim+1])/np.sum(s)
            dim += 1
        cluster_dim.append(dim)

    connection_dim = []
    for connection_act in connection_acts:
        _, s, _ = np.linalg.svd(connection_act)
        dim = 0
        var_explained = 0
        while var_explained < variance_limit:
            var_explained = np.sum(s[:dim+1]) / np.sum(s)
            dim += 1
        connection_dim.append(dim/(k-1)) # normalize by number of projections

    return np.array(cluster_dim), np.array(connection_dim)


if __name__ == '__main__':

    N = 300
    k = 6

    B_start = 10
    dt_delay = np.sqrt(8)/(B_start * propagation_vel)
    x_lim = [-1, 1]
    y_lim = [-1, 1]

    dt_no_delay = (np.sqrt(8) / propagation_vel) * 2


    dt = dt_delay


    results_dict = {
        'coco': {},
        'performance': {}
    }

    sr_ratio_grid = [.5, 1, 1.5, 2, 2.5, 3, 3.5]
    reps_per_setting = 5
    # sig_noise_ratio_grid = [1, 2, 4, 8, 16]
    for sr_ratio in sr_ratio_grid:
        print('Building starting network...')

        # net_params = params_start_evolution(N, dt=dt, k=k, x_lim=x_lim, y_lim=y_lim)

        net_params = get_comp_net(N, k, x_lim, y_lim, 0.5, 0.5*sr_ratio, .5, .5, dt)
        results_dict['coco'][sr_ratio] = []
        results_dict['performance'][sr_ratio] = []

        for i in range(reps_per_setting):
            start_net = populations.GMMPopulation(**net_params)
            sim = NetworkSimulator(start_net)
            sim.reset()
            sim.warmup(np.random.uniform(0, 1, 500))
            task_in = np.random.uniform(0, 1, 500)
            net_act = sim.get_network_data(task_in)
            # sim.visualize(task_in)

            intra, inter = get_coco(net_act, start_net.W, start_net.cluster_assignment, variance_limit=.98, D=start_net.D)

            data_train = np.array(utils.createNARMA30(8000)).reshape((2, 8000))
            data_val = np.array(utils.createNARMA30(4000)).reshape((2, 4000))


            performance = eval_net(start_net, data_train, data_val, alphas=[10e-7, 10e-5, 10e-3])
            results_dict['coco'][sr_ratio].append((intra, inter))
            results_dict['performance'][sr_ratio].append(performance)


    filename = "coco_tests/coco_performance_17042024_results_linear_scale.p"

    with open(filename, 'wb') as file:
        pkl.dump(results_dict, file)
        file.close()

