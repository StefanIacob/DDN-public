from simulator import NetworkSimulator
import pickle as pkl
import argparse
from reservoirpy.datasets import mackey_glass
import numpy as np

def get_example_networks(opt_path):
    with open(opt_path, "rb") as f:
        opt_dict = pkl.load(f)
    print(opt_dict.keys())
    example_net = opt_dict["example net"]
    pop_size = opt_dict["parameters"].shape[1]
    rand_i = np.random.randint(0, pop_size)
    random_params = opt_dict["parameters"][0, rand_i, :]
    best_params = opt_dict["evolutionary strategy"].best.x
    random_net = example_net.get_new_network_from_serialized(random_params)
    best_net = example_net.get_new_network_from_serialized(best_params)
    return random_net, best_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delay', action="store_true", help="Run experiment with delays",
                        default=True)
    args = parser.parse_args()
    delay = args.delay

    # Load optimized DDN
    # opt_path_ddn = "ADDN_further_experiments/Results N300 K5/2023-08-24_delay_True_bcm_False_growing_False.p"
    # opt_path_esn = "ADDN_further_experiments/Results N300 K5/2023-08-24_delay_False_bcm_False_growing_False.p"
    opt_path_ddn = "../NARMA-30_results_23/NARMA30_old_ddn_results_n101_k4_date_2023-12-11.p"
    opt_path_esn = "../NARMA-30_results_23/NARMA30_old_bl_results_n101_k4_date_2024-03-18.p"
    opt_path = opt_path_esn
    if delay:
        opt_path = opt_path_ddn

    random_net, best_net = get_example_networks(opt_path)

    # Generate input data
    example_data = mackey_glass(1000)
    random_data = np.random.uniform(low=0, high=0.5, size=1000)

    # Run simulation
    sim = NetworkSimulator(random_net)
    sim.visualize(random_data)
