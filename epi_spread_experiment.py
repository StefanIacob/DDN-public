import numpy as np
import populations
from scipy.special import softmax
from network import sigmoid_activation, tanh_activation, get_multi_activation, relu
from config import propagation_vel
from simulator import NetworkSimulator
from reservoirpy import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from sklearn import mixture
import os
from datetime import date
import argparse
from evolution import cmaes_signal_gen_adaptive_custom_datasets
from utils import eval_candidate_custom_data_signal_gen
from scipy import stats

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

def params_start_evolution_adaptive(N, dt, var_delays, k, size_in, net_size_factor=1):
    outsize = N - size_in
    x_lim = None
    y_lim = None
    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = .002 * net_size_factor
    height = .003 * net_size_factor

    # location
    start_mu = np.zeros((k, 2))
    for i in range(k):
        start_mu[i, :] = [(i+size_in) * (width/(k+size_in)), (i+size_in) * (height/(k+size_in))]

    var = np.ones((k, 2)) * width * 0.1
    corr = np.ones((k,)) * 0

    inhib_start = np.zeros((k, 1))
    in_loc = (0, 0)

    # connectivity and scaling indices as follows: from columns to rows. -1 is input
    connectivity = np.ones((k+size_in, k+size_in)) * .02
    weight_scaling = np.ones_like(connectivity) * 1.2
    lr_mean = np.ones_like(weight_scaling) * 0.01
    lr_scaling = np.ones_like(lr_mean) * 0.01
    weight_mean = np.zeros_like(weight_scaling)
    bias_mean = np.zeros((k+size_in, ))

    cluster_connectivity = np.ones((k + size_in, k + size_in))
    bias_scaling = np.ones((k + size_in,)) * .5
    y0_mean = np.ones_like(bias_mean) * .8
    y0_scaling = np.ones_like(bias_mean) * .1
    decay_start = np.ones((k + size_in,)) * .99
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
        'size_in': size_in,
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
    file_path_1 = "COVID-19_aantallen_gemeente_per_dag_tm_03102021.csv"
    file_path_2 = "COVID-19_rioolwaterdata_gemeentenweek.csv"
    file_path_3 = "COVID-19_aantallen_gemeente_per_dag.csv"
    file_path_4 = "COVID-19_rioolwaterdata.csv"
    file_path_5 = "COVID-19_rioolwaterdata_ggdregio.csv"
    file_path_osi = "stringency_index_nl.csv"

    # Load data
    folder = "task_data/RIVM_project/"
    data = pd.read_csv(folder + file_path_3, delimiter=';')

    # Load context variables
    stringency_data = pd.read_csv(folder + file_path_osi, delimiter=',')
    stringency_sequence = np.array(stringency_data["stringency_index"])

    location_index_name = 'Municipality_code'
    feature_name = 'Total_reported'

    region_codes = data[location_index_name].unique()
    data_dict = {}
    seq_len = 582
    for region_code in region_codes:
        if isinstance(region_code, str):
            region_specific_df = data.loc[data[location_index_name] == region_code]
            #     sequence.interpolate()

            sequence = np.array(region_specific_df[feature_name])
            #     print(len(sequence))
            #     plt.figure()
            #     plt.plot(sequence)
            #     plt.title(region_code)
            if len(sequence) >= seq_len:
                ROAZ_names = region_specific_df["ROAZ_region"].unique()
                #         display(region_specific_df)
                for i, ROAZ_name in enumerate(ROAZ_names):
                    ROAZ_specific_df = region_specific_df.loc[region_specific_df["ROAZ_region"] == ROAZ_name]
                    sequence = np.array(ROAZ_specific_df[feature_name])
                    sequence = np.stack([sequence, stringency_sequence])
                    #             plt.plot(sequence)
                    data_dict[region_code + ROAZ_name] = sequence.T
            else:
                sequence = np.stack([sequence, stringency_sequence])
                data_dict[region_code] = sequence.T
    #             print(len(sequence))

    print(data_dict.keys())

    region_codes = list(data_dict.keys())
    data_dict_clustered = {}
    data_dict_clustered_first = {}

    # Cluster highly correlated municipalities

    for i in region_codes:
        seq_1 = data_dict[i]
        data_dict_clustered[i] = [seq_1]
        data_dict_clustered_first[i] = seq_1
        for j in region_codes:
            seq_2 = data_dict[j]
            r, p = stats.pearsonr(seq_1[:, 0], seq_2[:, 0])
            if np.abs(r) > .8 and p < .05 and not (i == j):
                data_dict_clustered[i].append(seq_2)
                #             print(r)
                region_codes.remove(j)
    total = 0
    for key in data_dict_clustered:
        seq = data_dict_clustered[key]
        print(len(seq))
        total += len(seq)

    print('total length:')
    print(total)

    size_in = 2

    # create train/val sequences
    unsupervised = data_dict_clustered['GM0034']
    supervised = data_dict_clustered['GM0014']
    validation = data_dict_clustered['GM0047']

    N = 200 + size_in
    k = 6
    dt = .0005
    net_scaling = 25
    pop_size = 25
    print('Building starting network...')
    net_params = params_start_evolution_adaptive(N, dt, True, k, size_in=size_in, net_size_factor=net_scaling)

    start_net = populations.GMMPopulationAdaptive(**net_params)
    file_name = "cma_es_debugging_covid_cases" + str(date.today())

    cmaes_signal_gen_adaptive_custom_datasets(start_net, unsupervised, supervised, validation, pop_size, max_it=100,
                                                  dir='RIVM_es_results', name=file_name,
                                                  alphas=[10e-14, 10e-13, 10e-12], error_margin=.1,
                                                  fitness_function=None)
