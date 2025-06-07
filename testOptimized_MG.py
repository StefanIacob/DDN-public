import numpy as np
import pickle as pkl
from reservoirpy import datasets
from simulator import NetworkSimulator
from utils import single_sample_NRSE, eval_candidate_signal_gen_horizon
from datetime import date
import argparse


def resample_net_MG_best(data_dict, maxgen=None):
    validation_scores = data_dict['validation performance']
    if not maxgen is None:
        validation_scores = validation_scores[:maxgen, :, :]
    score = np.mean(validation_scores, axis=-1)
    max_gens = np.max(score, axis=-1)
    best_gen = np.argmax(max_gens)
    best_pop = score[best_gen, :]
    best_ind = np.argmax(best_pop)
    all_params = data_dict['parameters']
    best_params = all_params[best_gen, best_ind]
    net = data_dict['example net']
    best_net = net.get_new_network_from_serialized(best_params)
    return best_net

def resample_net_MG_worst(data_dict):
    validation_scores = data_dict['validation performance']
    score = np.min(validation_scores, axis=-1)
    max_gens = np.max(score, axis=-1)
    best_gen = np.argmax(max_gens)
    best_pop = score[best_gen, :]
    best_ind = np.argmax(best_pop)
    all_params = data_dict['parameters']
    # best_params = all_params[100, 8]
    best_params = all_params[best_gen, best_ind]
    net = data_dict['example net']
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

    for sequence in test_data:
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
    parser.add_argument("-g", "--maxgen", action="store", type=int, default=None, help="Takes best up to this generation")


    args = parser.parse_args()
    config = vars(args)
    resamples = config['resamples']
    n_test_samples = config['testsamples']
    n_test_sequences = config['testsequences']
    path = config['path']
    maxgen = config['maxgen']
    # Load data
    print("Loading hyperparameter optimization results from " + path)

    with open(path, 'rb') as f:
        results_dict = pkl.load(f)

    tau_list = results_dict['tau list'] # get tau range from any of the results dict
    x0_range = results_dict['start value range']

    # Generate test data
    # n_test_samples = 502
    test_data_tau = {}
    warmup = 400
    for tau in tau_list:
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
        # best_net = resample_net_MG_worst(results_dict)
        # testVisualize(best_net, test_data_tau[14][0])
        best_net = resample_net_MG_best(results_dict, maxgen=maxgen)
        resampled_networks.append(best_net)

    for tau in tau_list:
        test_results[tau] = []
        print("Testing for tau = " + str(tau))
        error_margin = results_dict['error margin']
        for resample, net in enumerate(resampled_networks):
            print("Resample " + str(resample))
            val, model, net = retrain_net_MG(net, results_dict, tau)
            _, t_performance = test_net_MG(net, model, error_margin, test_data_tau[tau])
            test_results[tau].append(t_performance)

    save_path = path[:-2] + '_gen' + str(maxgen) + '_test_optimized.p'
    with open(save_path, 'wb') as f:
        pkl.dump(test_results, f)