import numpy as np
import pickle as pkl
from reservoirpy import datasets
from simulator import NetworkSimulator
from utils import single_sample_NRSE, eval_candidate_signal_gen_multiple_random_sequences_adaptive
from datetime import date
import argparse


def retrain_net(data_dict, tau):
    best_params = data_dict['evolutionary strategy'].best.x
    net = data_dict['example net']
    best_net = net.get_new_network_from_serialized(best_params)
    x0_range = data_dict['start value range']
    n_seq = data_dict['number of sequences']
    n_sam = data_dict['number of samples']
    tau_range = [tau, tau]
    error_margin = data_dict['error margin']
    alphas = data_dict['alpha grid']
    val, model, net = eval_candidate_signal_gen_multiple_random_sequences_adaptive(best_net,
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

def test_net(network, model, error_margin, test_data):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--resamples", action="store", type=int, default=100, help="Number of resamples per test")

    args = parser.parse_args()
    config = vars(args)
    resamples = config['resamples']
    evolution_data = {
        'ESN': {
            'fixed': {
                'path': "results-2023-paper/2022-12-08_b1_n300_k6_multiple_sequences_random_tau_adaptive_False_delays_False.p"},
            'BCM': {
                'path': "results-2023-paper/2022-12-05delay_BCM_b25_n300_k6_multiple_sequences_BCM_5_5_seq_random_tau.p"}
        },
        'DDN': {
            'fixed': {
                'path': "results-2023-paper/2022-12-06delay_BCM_b25_n300_k6_multiple_sequences_BCM_5_5_seq_random_tau.p"},
            'BCM': {
                'path': "results-2023-paper/2022-12-02delay_BCM_b25_n300_k6_multiple_sequences_BCM_5_5_seq_random_tau.p"}
        }
    }
    # Load data
    for net_type in evolution_data:
        for adaptive in evolution_data[net_type]:

            path = evolution_data[net_type][adaptive]['path']
            print("Loading hyperparameter optimization results from " + path)

            with open(path, 'rb') as f:
                results_dict = pkl.load(f)
            evolution_data[net_type][adaptive]['results_dict'] = results_dict

    tau_range = results_dict['tau range'] # get tau range from any of the results dict
    x0_range = results_dict['start value range']

    # Generate test data
    n_test_samples = 502
    test_data_tau = []
    warmup = 400
    for tau in range(tau_range[0], tau_range[1] + 1):
        test_data_sequences = []
        for i in range(resamples):
            test_data = datasets.mackey_glass(n_test_samples + warmup, tau=tau,
                                              x0=np.random.uniform(x0_range[0], x0_range[1]))
            test_data_sequences.append(test_data)
        test_data_tau.append(test_data_sequences)

    test_results = {}
    for net_type in evolution_data:
        test_results[net_type] = {}
        for adaptive in evolution_data[net_type]:
            test_results[net_type][adaptive] = {}
            print('Testing ' + net_type + ' with ' + adaptive + ' weigths')
            results_dict = evolution_data[net_type][adaptive]['results_dict']
            for tau in range(tau_range[0], tau_range[1]+1):
                test_results[net_type][adaptive][tau] = []
                print("Testing for tau = " + str(tau))
                error_margin = results_dict['error margin']
                for resample in range(resamples):
                    print("Resample " + str(resample))
                    val, model, net = retrain_net(results_dict, tau)
                    _, t_performance = test_net(net, model, error_margin, test_data_tau[tau - 12])
                    print(t_performance)
                    test_results[net_type][adaptive][tau].append(np.mean(t_performance))

    filename = "random_tau_BCM_test_results_" + str(date.today()) + ".p"
    path = "results-2023-paper/" + filename
    with open(path, 'wb') as f:
        pkl.dump(test_results, f)