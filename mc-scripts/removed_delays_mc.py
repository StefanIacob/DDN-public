import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import pickle as pkl
import numpy as np
import argparse
from network import DistDelayNetwork
from utils import network_memory_capacity
from datetime import date

def strip_delays(evolved_ddn):
    removed_delays = DistDelayNetwork(evolved_ddn.W, evolved_net.WBias, evolved_net.n_type, np.zeros_like(evolved_net.coordinates),
                                      evolved_net.decay, evolved_net.neurons_in, evolved_net.neurons_out,
                                      evolved_net.activation_func, evolved_net.dt, 0, evolved_net.theta_y0,
                                      np.max(evolved_net.lr))
    return removed_delays

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
    parser.add_argument("output_folder", type=str, help="output file path")

    args = parser.parse_args()
    filename = args.input_filename
    output_path = args.output_folder

    # Load dict
    print("loading file: " + filename)

    results_dict = load_file(filename)

    alphas = [10e-7, 10e-5, 10e-3]
    if "alpha grid" in results_dict.keys():
        alphas = results_dict["alpha grid"]

    # Generate Networks
    print("generating best network")
    ex_net = results_dict['example net']
    best_parameters = results_dict['evolutionary strategy'].best.x
    all_parameters = results_dict['parameters']
    evolved_net = ex_net.get_new_network_from_serialized(best_parameters)
    print("generating delay-stripped network")
    removed_delays_net = strip_delays(evolved_net)

    # Compute MC
    max_del = 70
    n_reps = 20
    print("Computing MC of evolved network for " + str(n_reps) + " repetitions up to delay " + str(max_del))
    caps_evolved_net = network_memory_capacity(evolved_net, max_del, 1000, None,
                                               400, alphas, reps=n_reps)
    print("Computing MC of delay-stripped network for " + str(n_reps) + " repetitions up to delay " + str(max_del))
    caps_removed_delays = network_memory_capacity(removed_delays_net, max_del, 1000, None,
                                                  400, alphas, reps=n_reps)

    # Save MC
    print("Saving mc files: ")
    evolved_net_path = output_path + "/evolved_net_" + str(date.today()) + ".p"
    removed_delays_net_path = output_path + "/removed_delays_" + str(date.today()) + ".p"
    print(evolved_net_path)
    print(removed_delays_net_path)

    save_file(evolved_net_path, np.array(caps_evolved_net))
    save_file(removed_delays_net_path, np.array(caps_removed_delays))
