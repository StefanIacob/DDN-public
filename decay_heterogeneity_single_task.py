from datetime import date
import argparse
import os
from populations import FlexiblePopulation
from simulator import NetworkSimulator
import numpy as np
from evolution import cmaes_alg_gma_pop_timeseries_prediction_old
from utils import createNARMA30
from config import propagation_vel, get_p_dict_heterogeneity_exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-k", "--clusters", action="store", help="number of GMM clusters to be used",
                        type=int, default=5)
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-dd", "--distributed_decay", action="store_true", help="Distributed decay parameter")
    parser.add_argument("-cd", "--cluster_decay", action="store_true", help="Different decay per cluster")


    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    N = config['neurons']
    K = config['clusters']
    distributed_decay = config['distributed_decay']
    per_cluster_decay = config['cluster_decay']

    max_delay = 0.1
    if delay:
        max_delay = 12
    x_range = [-.01, .01]
    y_range = [-.01, .01]
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width**2 + height**2)
    max_time = max_dist/propagation_vel
    dt = max_time/max_delay
    in_loc = (-1, 1)
    size_in = 1
    size_out = N - size_in
    p_dict = get_p_dict_heterogeneity_exp(K, x_range, y_range)

    if not distributed_decay:
        # If not distributed, a fixed decay value per cluster or network should be used
        # Hence scaling is fixed to 0
        p_dict['decay_scaling']['val'] = np.array([0])
        p_dict['decay_scaling']['evolve'] = False

    if per_cluster_decay:
        # use different decay per cluster, which means decay parameter should be of size K
        p_dict['decay_mean']['val'] = np.array(K * list(p_dict['decay_mean']['val']))
        p_dict['decay_scaling']['val'] = np.array(K * list(p_dict['decay_scaling']['val']))


    start_net = FlexiblePopulation(N, x_range, y_range, dt, in_loc, size_in, size_out,
                     p_dict)

    data_train = np.array(createNARMA30(8000)).reshape((2, 8000))
    data_val = np.array(createNARMA30(4000)).reshape((2, 4000))
    gens = 2#200
    pop_size = 3 #20
    reps_per_cand = 2#5
    alphas = [10e-7, 10e-5, 10e-3]
    dir='heterogeneity_results'
    filename='pilot'
    cmaes_alg_gma_pop_timeseries_prediction_old(start_net, data_train, data_val, gens, pop_size, reps_per_cand, .3,
                                                alphas, dir=dir, name=filename)
