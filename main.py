import numpy as np
import pickle
import populations
import evolution
from scipy.special import softmax
import matplotlib.pyplot as plt
from network import sigmoid_activation, tanh_activation
from config import propagation_vel
from utils import createNARMA10
from simulator import NetworkSimulator
import utils
import seaborn as sns
from datetime import date

if __name__ == '__main__':
    N = 301
    insize = 1
    outsize = N - insize
    # np.random.seed(42)
    data_train = np.array(createNARMA10(8000)).reshape((2, 8000))
    data_val = np.array(createNARMA10(4000)).reshape((2, 4000))
    data_test = np.array(createNARMA10(4000)).reshape((2, 4000))
    x_range = (0, .002)
    y_range = (0, .004)
    k = 4
    max_it = 100
    pop_size = 25
    std = 0.1

    start_mix = np.ones((k,))
    start_mix = softmax(start_mix)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    start_mu = np.zeros((k, 2))
    start_mu[:, :] = (np.mean(x_range) + 0.2 * width, np.mean(y_range) + 0.4 * width)
    start_var = (width) * 0.15
    start_corr = np.ones((k,)) * 0.15
    start_var = np.random.uniform(start_var, start_var, (k, 2))
    inhib_start = np.ones((k,)) * 0
    in_loc = (x_range[0] + width * 0.3, y_range[0] + height * 0.2)
    conn_start = np.ones((k+1, k+1))
    # conn_start = np.array(
    #     [
    #         [0.5, 0.5],
    #         [0, 0.5]
    #     ]
    # )
    # conn_start = np.array(
    #     [
    #         [0.5, 0.5, 0.5],
    #         [0.5, 0.5, 0.5],
    #         [0.5, 0.5, 0.5]
    #     ]
    # )

    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling_start = np.ones((k + 1,)) * 0.5
    weight_scaling_start = np.ones((k + 1, k + 1)) * .3
    weight_scaling_start[0, 1] = 0.9
    decay_start = np.ones((k + 1,)) * .95
    dt = .5
    buffersize = 1
    activation = tanh_activation
    # activation = sigmoid_activation
    # alphas = [10e-14, 10e-13, 10e-12]
    alphas = [10e-7, 10e-5, 10e-3]
    # alphas = [10e-6, 10e-5, 10e-4]
    # alphas = [0]
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
        'x_range': x_range,
        'y_range': y_range,
        'decay': decay_start,
        'size_in': insize,
        'size_out': outsize,
        'in_loc': in_loc,
        'act_func': activation,
        'dt': dt,
        'buffersize': buffersize
    }
    net = populations.GMMPopulationOld(**net_params)

    # from netPropertyTools import impulse_response
    # act = impulse_response(net, mag=100, rec_time=50, times=10, pre_pulse=30, visual=False)
    # for i in range(10):
    #     rel_act = act[:, 30 + (i) * 50:30 + (i+1) * 50]
    #     resp = np.zeros((50,))
    #     for j in range(50):
    #         resp[j] = rel_act[:, j] @ rel_act[:, j].T
    #     plt.plot(resp)
    #     plt.show()
    # file = open("vd_net_to_test_sigmoid.p", "wb")
    # pickle.dump(net, file)
    # net.get_serialized_parameters()

    # i_gen = 16
    # i_pop = 24
    # filename = "es_results/cma_es_gmm_k1_tanh_final_paper_bl2022-10-26.p"
    # file = open(filename, "rb")
    # data = pickle.load(file)
    # last_gen_val = data['validation performance'][i_gen]
    # last_gen_val = np.min(np.mean(last_gen_val, axis=1), axis=1)
    # # i_pop = np.argmin(last_gen_val)
    # params = data['parameters']
    # serialized_sample = params[i_gen, i_pop, :]
    # net = net.get_new_network_from_serialized(serialized_sample)

    # performance = perfs[i_gen, i_pop, :, :]
    # # plt.figure(figsize=(10,5))
    # # sns.heatmap(performance, annot=True)
    # # plt.show()

    # print(net.weight_scaling)
    # sim = NetworkSimulator(net)
    # sim.visualize(data_train[0, :])

    # val_perfs = []
    # lags = []

    # for i in range(10):
    #     net = net.get_new_network_from_serialized(check_par)
    #     t, v, m = utils.eval_candidate_lag_gridsearch(net, data_train, data_test, warmup=400)
    #     print(t, v)
    #     best_lag = np.argmin(v)
    #     print(best_lag, v[best_lag])
    #     val_perfs.append(v[best_lag])
    #     lags.append(best_lag)


    print(net.activation_func)
    dir = 'es_results'
    name = 'cma_es_gmm_k4_tanh_final_paper_bl' + str(date.today())
    print('dt', dt)
    print('B', buffersize)
    print('Regularization', alphas)
    print('k', k)
    evolution.cmaes_alg_gma_pop_timeseries_prediction(net, data_train, data_val, max_it, pop_size, std=std, dir=dir, name=name, alphas=alphas)
