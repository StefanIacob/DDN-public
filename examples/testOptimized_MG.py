import numpy as np
import pickle as pkl
from reservoirpy import datasets
from simulator import NetworkSimulator
from utils import single_sample_NRSE, eval_candidate_signal_gen_horizon
from datetime import date
import argparse
from populations import GMMPopulationAdaptive, GMMPopulationOld, GMMPopulationAdaptiveOld
from tqdm import tqdm

def resample_net_MG_best(data_dict):
    best_params = data_dict['evolutionary strategy'].best.x
    net = data_dict['example net']
    net = cast_to_old(net)
    best_net = net.get_new_network_from_serialized(best_params)
    return best_net

def retrain_net_MG(best_net, data_dict, tau):
    best_net.reset_network()
    x0_range = data_dict['start value range']
    n_seq = data_dict['number of sequences']
    n_sam = data_dict['number of samples']
    tau_range = [tau, tau]
    error_margin = data_dict['error margin']
    alphas = data_dict['alpha grid']
    val, model, net = eval_candidate_signal_gen_horizon(best_net,
                                                          n_seq['unsupervised'],
                                                          n_seq['supervised'],
                                                          0,
                                                          n_sam['unsupervised'],
                                                          n_sam['supervised'],
                                                          0,
                                                          error_margin=error_margin,
                                                          alphas=alphas,
                                                          tau_range=tau_range,
                                                          x0_range=x0_range
                                                         )
    return val, model, net

def test_net_MG(network, model, error_margin, test_data):
    warmup = 400
    prediction_steps_across_sequences = []
    y_across_sequences = []
    max_it_val = 500
    sim = NetworkSimulator(network)

    for sequence in tqdm(test_data):
        start_input_val = sequence[warmup]
        labels_val = sequence[warmup + 1:]
        sim.warmup(sequence[:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val
        label_variance = np.var(labels_val)
        steps = 0
        y = []
        while j <= max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            if len(output.shape) == 1:
                output = np.expand_dims(output, 0)
            feedback_in = model.predict(output)[0][0]
            y.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            j += 1
            if error <= error_margin:
                steps += 1

        prediction_steps_across_sequences.append(steps)
        y_across_sequences.append(y)
    return y_across_sequences, prediction_steps_across_sequences

def cast_to_old(network):
    """
        Helper function to load the older pickled versions of GMMPopulationAdaptive to GMMPopulationAdaptiveOld
    """
    if type(network) == GMMPopulationAdaptive:
        network.__class__ = GMMPopulationAdaptiveOld
    return network


def testVisualize(network, data):
    sim = NetworkSimulator(network, False)
    sim.visualize(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", action="store", type=str, default="./", help="Evolution results data path")
    parser.add_argument("-r", "--resamples", action="store", type=int, default=100, help="Number of resamples per test")
    parser.add_argument("-t", "--testsamples", action="store", type=int, default=502, help="Test sequence length")
    parser.add_argument("-s", "--testsequences", action="store", type=int, default=5, help="Number of test sequences per network")


    args = parser.parse_args()
    config = vars(args)
    resamples = config['resamples']
    n_test_samples = config['testsamples']
    n_test_sequences = config['testsequences']
    path = config['path']

    # Load data
    print("Loading hyperparameter optimization results from " + path)

    with open(path, 'rb') as f:
        results_dict = pkl.load(f)

    tau_range = results_dict['tau range'] # get tau range from any of the results dict
    x0_range = results_dict['start value range']

    # Generate test data
    test_data_tau = {}
    warmup = 400
    for tau in range(tau_range[0], tau_range[1] + 1):
        test_data = []
        for seq in range(n_test_sequences):
            test_sequence = datasets.mackey_glass(n_test_samples + warmup, tau=tau,
                                              x0=np.random.uniform(x0_range[0], x0_range[1]))
            test_data.append(test_sequence)
        test_data_tau[tau] = test_data

    test_results = {}

    resampled_networks = []
    print("Sample networks")

    for resample in range(resamples):
        best_net = resample_net_MG_best(results_dict)
        resampled_networks.append(best_net)

    for tau in range(tau_range[0], tau_range[1]+1):
        test_results[tau] = []
        print("Testing for tau = " + str(tau))
        error_margin = results_dict['error margin']
        for resample, net in enumerate(tqdm(resampled_networks)):
            print("Resample " + str(resample))
            val, model, net = retrain_net_MG(net, results_dict, tau)
            _, t_performance = test_net_MG(net, model, error_margin, test_data_tau[tau])
            print(t_performance)
            test_results[tau].append(t_performance)

    save_path = path[:-2] + '_test_optimized.p'
    with open(save_path, 'wb') as f:
        pkl.dump(test_results, f)