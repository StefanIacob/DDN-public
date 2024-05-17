import numpy as np
import populations
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation, get_multi_activation, relu
from config import propagation_vel
from simulator import NetworkSimulator, NetworkArmSimulator
from arm_model import Arm2Link
from reservoirpy import datasets
from evolution import cmaes_alg_gma_pop_signal_gen, cmaes_alg_gma_pop_signal_gen_adaptive, continue_cmaes_adaptive
from datetime import date
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from sklearn import mixture
import argparse
import pickle as pkl
from gui import DistDelayGUI

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

def params_start_evolution_adaptive(N, var_delays, k, x_lim=None, y_lim=None, dt=.0005, insize=4, outsize=2):
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

    var = np.ones((k, 2)) * width * 0.3
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k, 1))
    in_loc = np.array([(x_range[0] + width * 0.01 * i, y_range[0] + height * 0.01) for i in range(insize)])

    out_loc = np.array([(x_range[1] - width * 0.01 * i, y_range[1] - height * 0.01) for i in range(outsize)])

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+1, k+1)) * .2
    weight_scaling = np.ones_like(connectivity) * 1
    lr_mean = np.ones_like(weight_scaling) * 0.5
    lr_scaling = np.ones_like(lr_mean) * 0.05
    weight_mean = np.ones_like(weight_scaling) * -0.1
    bias_mean = np.zeros((k+1, ))

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.ones((k + 1,)) * .2
    y0_mean = np.ones_like(bias_mean) * .9
    y0_scaling = np.ones_like(bias_mean) * .1
    decay_start = np.ones((k + 1,)) * .9

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
        'out_loc': out_loc,
        'act_func': activation,
        'dt': dt,
        # 'theta_window': None,
        'var_delays': var_delays
    }
    return net_params

def params_start_evolution(N, dt, k, x_lim=None, y_lim=None, insize=6, outsize=2):
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
    in_loc = np.array([(x_range[0] + width * 0.1 * i, y_range[0] + height * 0.01) for i in range(insize)])

    out_loc = np.array([(x_range[1] - width * 0.1 * i, y_range[1] - height * 0.01) for i in range(outsize)])

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+1, k+1)) * .3
    weight_scaling = np.ones_like(connectivity) * .5
    weight_mean = np.zeros_like(weight_scaling) * -0.1
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
        'out_loc': out_loc,
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


    # mg_train = mackey_glass[:10000]
    # mg_train = np.random.uniform(size=(10000,))
    # mg_validate = mackey_glass[10000:]
    delay = True
    adaptive = False
    N = 150
    k = 4
    B_start = 25
    dt_delay = np.sqrt(8)/(B_start * propagation_vel)
    x_lim = [-1, 1]
    y_lim = [-1, 1]

    dt_no_delay = (np.sqrt(8) / propagation_vel) * 2

    x_lim = y_lim = None

    dt = dt_delay
    if not delay and not adaptive:
        dt = dt_no_delay

    print('Building starting network...')

    # if adaptive:
    #     net_params = params_start_evolution_adaptive(N=N, dt=dt, x_lim=x_lim, y_lim=y_lim, var_delays=delay, k=k)
    #     start_net = populations.GMMPopulationAdaptive(**net_params)
    # else:
    net_params = params_start_evolution(N, dt=dt, k=k, x_lim=x_lim, y_lim=y_lim)
    start_net = populations.ArmPopulation(**net_params)

    arm = Arm2Link(dt=dt)

    sim = NetworkArmSimulator(start_net, arm)
    targets = np.repeat([[-0.5, 1.1]], 10000, axis=0)
    sim.visualize_arm(targets)
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

