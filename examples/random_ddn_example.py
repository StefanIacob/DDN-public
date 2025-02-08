from datetime import date, datetime
import argparse
from populations import FlexiblePopulation
import numpy as np
from simulator import NetworkSimulator
from utils import createNARMA30, createNARMA10
from config import propagation_vel, get_p_dict_heterogeneity_exp, get_p_dict_like_p3_NARMA
from network import tanh_activation

if __name__ == '__main__':
    N = 50  # Number of neurons
    K = 3  # Number of clusters
    use_delays = True  # True for DDNs, False for ESNs
    x_range = [-.01, .01] # x spatial dimension ranges
    y_range = [-.01, .01] # y spatial dimension ranges
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width ** 2 + height ** 2)
    max_time = max_dist / propagation_vel
    max_delay = 0.001
    if use_delays:
        max_delay = 20 # Maximum desired delay in case of DDN
    dt = max_time / max_delay
    in_loc = (-1, 1)  # Location of input neuron
    size_in = 1
    size_out = N - size_in
    p_dict = get_p_dict_like_p3_NARMA(K, x_range, y_range)  # standard network configuration.
    # Play around with parameters

    activation_func = tanh_activation
    start_net = FlexiblePopulation(N, x_range, y_range, dt, in_loc, size_in, size_out,
                     p_dict, act_func=activation_func)

    # Input data
    uniform_in = np.random.uniform(0, 0.5, (1000,))

    sim = NetworkSimulator(start_net)

    # ----- Network simulation with GUI -----
    # sim.visualize(uniform_in)
    # ---------------------------------------


    # ------ Simulating and saving network data ------
    # network_activity = sim.get_network_data(uniform_in)
    # ------------------------------------------------

    # ------ Training and evaluating readout layer for NARMA 10 task ------
    # data_train = np.array(createNARMA10(8000)).reshape((2, 8000))
    # data_val = np.array(createNARMA10(4000)).reshape((2, 4000))
    # data_test = np.array(createNARMA10(4000)).reshape((2, 4000))
    # from utils import eval_candidate_lag_gridsearch_NARMA
    # # trains readouts for a number of lags, returns train scores, validation scores, and readout models for each lag
    # # train scores and validation scores are given as NRMSE, smaller is better
    # alphas = [10e-7, 10e-5, 10e-3]
    # train_scores_lag, val_scores_lag, models_lag = eval_candidate_lag_gridsearch_NARMA(start_net,
    #                                                                                    data_train,
    #                                                                                    data_val,
    #                                                                                    alphas=alphas)
    # lag = np.argmin(val_scores_lag)  # determine best lag
    # readout = models_lag[lag]  # select best model
    #
    # # Make predictions with readout layer
    # sim.reset()
    # sim.warmup(data_test[0,:400])
    # test_act = sim.get_network_data(data_test[0,400:])
    # predictions = readout.predict(test_act.T)
    # # Plot predictions vs labels
    # import matplotlib.pyplot as plt
    # plot_start = 100
    # plot_end = 150
    #
    # # Make sure to take lag into account, predictions are shifted back by lag timesteps
    # plt.plot(predictions[plot_start:plot_end], label='prediction')
    # plt.plot(data_test[1, 400 + plot_start - lag:400 + plot_end - lag], label='label')
    # plt.legend()
    # plt.show()
    # ----------------------------------------------------------------
