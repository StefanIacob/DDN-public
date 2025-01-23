import pickle as pkl
import numpy as np
import argparse
from network import DistDelayNetwork
from utils import network_memory_capacity

def strip_delays(evolved_ddn):
    removed_delays = DistDelayNetwork(evolved_ddn.W, evolved_net.WBias, evolved_net.n_type, evolved_net.coordinates,
                                      evolved_net.decay, evolved_net.neurons_in, evolved_net.neurons_out,
                                      evolved_net.activation_func, evolved_net.dt, evolved_net.theta_window, evolved_net.theta_y0,
                                      np.max(evolved_net.lr))
    return removed_delays

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", type=str, help="input file path")
    parser.add_argument("output_folder", type=str, help="output file path")

    args = parser.parse_args()
    filename = args.input_filename
    output_path = args.output_folder

    # Load dict
    print("loading file: " + filename)
    with open(filename, 'rb') as f:
        results_dict = pkl.load(f)

    alphas = [10e-7, 10e-5, 10e-3]
    if "alpha grid" in results_dict.keys():
        alphas = results_dict["alpha grid"]

    # Generate Networks
    ex_net = results_dict['example net']
    best_parameters = results_dict['evolutionary strategy'].best.x
    all_parameters = results_dict['parameters']
    evolved_net = ex_net.get_new_network_from_serialized(best_parameters)
    removed_delays_net = strip_delays(evolved_net)

    # Compute MC
    caps_evolved_net = network_memory_capacity(evolved_net, 150, 1000, None,
                                               400, alphas, reps=20)
    caps_removed_delays = network_memory_capacity(removed_delays_net, 150, 1000, None,
                                                  400, alphas, reps=20)

    # Save MC
    print("Saving mc files: ")
    evolved_net_path = output_path + "/evolved_net.p"
    removed_delays_net_path = output_path + "/removed_delays.p"
    print(evolved_net_path)
    print(removed_delays_net_path)

    with open(evolved_net_path, 'wb') as f:
        pkl.dump(evolved_net, f)
    with open(removed_delays_net_path, 'wb') as f:
        pkl.dump(removed_delays_net, f)
