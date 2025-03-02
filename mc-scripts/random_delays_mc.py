import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import pickle as pkl
import numpy as np
import argparse
from populations import GMMPopulation, GMMPopulationOld, GMMPopulationAdaptive
from utils import memory_capacity
from config import propagation_vel
from network import DistDelayNetwork
from datetime import date

def esn_to_random_delay_ddn(evolved_esn, in_loc):
    k = evolved_esn.k

    mix = np.ones((k,))
    mix = mix / np.sum(mix)

    var = np.ones((k, 2)) * 0.1
    corr = np.ones((k,)) * 0
    mu = np.zeros((k, 2))
    if type(evolved_esn) == GMMPopulation:
        added_delays_net = GMMPopulation(evolved_esn.N, mix, mu, var, corr, evolved_esn.inhibitory,
                                         evolved_esn.connectivity, evolved_esn.cluster_connectivity,
                                         evolved_esn.weight_scaling, evolved_esn.weight_mean,
                                         evolved_esn.bias_scaling, evolved_esn.bias_mean,
                                         evolved_esn.decay, evolved_esn.size_in, evolved_esn.size_out,
                                         in_loc, evolved_esn.activation_func,
                                         evolved_esn.autapse,
                                         .0005, evolved_esn.x_lim, evolved_esn.y_lim,
                                         evolved_esn.fixed_delay)

    elif type(evolved_esn) == GMMPopulationOld:
        added_delays_net = GMMPopulationOld(N=evolved_esn.N, mix=mix, mu=mu, variance=var, correlation=corr,
                                            inhibitory=evolved_esn.inhibitory,
                                            connectivity=evolved_esn.connectivity,
                                            cluster_connectivity=evolved_esn.cluster_connectivity,
                                            weight_scaling=evolved_esn.weight_scaling,
                                            bias_scaling=evolved_esn.bias_scaling,
                                            decay=evolved_esn.decay, size_in=evolved_esn.size_in,
                                            size_out=evolved_esn.size_out,
                                            in_loc=in_loc, act_func=evolved_esn.activation_func,
                                            autapse=evolved_esn.autapse, dt=.0005,
                                            x_range=evolved_esn.x_range, y_range=evolved_esn.y_range,
                                            fixed_delay=evolved_esn.fixed_delay)

    elif type(evolved_esn) == GMMPopulationAdaptive: # Assumes unsupervised training has been done already
        added_delays_net = GMMPopulationAdaptive(evolved_esn.N, mix, mu, var, corr, evolved_esn.inhibitory,
                                                 evolved_esn.connectivity, evolved_esn.cluster_connectivity,
                                                 evolved_esn.weight_scaling, evolved_esn.weight_mean,
                                                 evolved_esn.bias_scaling, evolved_esn.bias_mean,
                                                 evolved_esn.decay, evolved_esn.lr_mean, evolved_esn.lr_scaling,
                                                 evolved_esn.y0_mean, evolved_esn.y0_scaling, evolved_esn.size_in,
                                                 evolved_esn.size_out, in_loc, evolved_esn.act_func, dt=0.0005)
    else:
        raise TypeError('evolved_esn should be GMMPopulation, GMMPopulationOld, or GMMPopulationAdaptive')

    return added_delays_net

def load_file(path):
    try:
        with open(path, "rb") as f:
            return pkl.load(f)
    except FileNotFoundError:
        with open("../" + path, "rb") as f:
            return pkl.load(f)


def save_file(path, data):
    try:
        with open(path, 'wb') as f:
            pkl.dump(data, f)
    except FileNotFoundError:
        with open("../" + path, "wb") as f:
            pkl.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", type=str, help="input file path")
    parser.add_argument("output_filename", type=str, help="output folder path")
    parser.add_argument("-ds", "--delay_step", type=int, default=25,
                        action="store", help="Stepsize for each increase in network scaling factor")
    parser.add_argument("-s", "--steps", type=int, default=5,
                            action="store", help="Number of network size increases")
    parser.add_argument("-id", "--input_distance", type=float, default=0, action="store",
                        help="Distance between input neuron and centre of reservoir")

    args = parser.parse_args()
    filename = args.input_filename
    output_path = args.output_filename

    delay_step = args.delay_step
    steps = args.steps
    input_distance = args.input_distance

    input_distance_str = "inDist_" + str(input_distance) + "_"
    output_path_delays_added = output_path + "/random_delays_added_" + input_distance_str + str(date.today()) + '.p'


    # Load dict
    print("loading file: " + filename)
    results_dict = load_file(filename)

    alphas = [10e-7, 10e-5, 10e-3]
    if "alpha grid" in results_dict.keys():
        alphas = results_dict["alpha grid"]

    # Generate Networks
    ex_net = results_dict['example net']
    best_parameters = results_dict['evolutionary strategy'].best.x
    all_parameters = results_dict['parameters']
    evolved_esn = ex_net.get_new_network_from_serialized(best_parameters)

    # Make a DDN with the exact same parameters as the DDN, with random delays
    in_loc = (-input_distance, 0)
    added_delays_net = esn_to_random_delay_ddn(evolved_esn, in_loc)

    covar_res = added_delays_net.covariances[0]
    var_res = np.max(np.linalg.eigvals(covar_res))
    relative_var = input_distance/var_res
    print("You have chosen an input distance of " + str(input_distance) + ". This corresponds to " + str(np.round(relative_var * 100, 2)) + "% of the reservoir variance.")
    # Loop through different network size scaling and compute mc
    d_max = np.max(added_delays_net.spatial_dist_continuous)
    mc_delays_added = []
    for i in range(steps):
        scaling = i * delay_step / (d_max/(added_delays_net.dt*propagation_vel))
        if i < 1:
            scaling = 0.0001

        increasing_delay_net = DistDelayNetwork(evolved_esn.W, evolved_esn.WBias, evolved_esn.n_type,
                                                added_delays_net.coordinates * scaling, evolved_esn.decay,
                                                evolved_esn.neurons_in, evolved_esn.neurons_out,
                                                evolved_esn.activation_func, added_delays_net.dt)
        print("scaled to a maximum delay of: " + str(np.max(increasing_delay_net.D)))


        mc = memory_capacity(increasing_delay_net, 150, 1000, z_function=None,
                        warmup_time=400, alphas=alphas)
        mc_delays_added.append(mc)

    save_file(output_path_delays_added, mc_delays_added)
