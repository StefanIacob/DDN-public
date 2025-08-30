import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import numpy as np
from network import DistDelayNetwork, tanh_activation
from utils import eval_candidate_lag_gridsearch_NARMA, createNARMA30, region_specific_IPC, IPC_overlap, createNARMA10
import pickle as pkl
from datetime import date

def write_result(results, sr, leak, task_p, optimal_lag, generation):
    results['sr'] = sr
    results['leak'] = leak
    results['task_p'] = task_p
    results['optimal_lag'] = optimal_lag
    results['generation'] = generation

def evaluate_networks(example_net, params_best_i, samples, train_data, val_data, results, generation):
    for sample in range(samples):
        opt_net = example_net.get_new_network_from_serialized(params_best_i)
        W = example_net.W
        v = np.linalg.eigvals(W)
        sr = np.max(np.absolute(v))
        av_leak = np.mean(opt_net.decay)
        _, val_performance_per_lag, _ = eval_candidate_lag_gridsearch_NARMA(opt_net, train_data, val_data, alphas=[10e-7, 10e-5, 10e-3])
        optimal_lag = np.argmin(val_performance_per_lag)
        print('optimal lag =', optimal_lag)
        print('test performance: ', val_performance_per_lag[optimal_lag])
        write_result(results, sr, av_leak, optimal_lag, sample, generation)

if __name__ == "__main__":
    # data_train = np.array(createNARMA30(4000)).reshape((2, 4000))
    # data_val = np.array(createNARMA30(6000)).reshape((2, 6000))
    data_train = np.array(createNARMA10(4000)).reshape((2, 4000))
    data_val = np.array(createNARMA10(6000)).reshape((2, 6000))
    samples = 3

    results = {
        'sr': [],
        'leak': [],
        'task_p': [],
        'optimal_lag': [],
        'generation': []
    }

    # filepath_results = 'results/NARMA-30_results_23/NARMA30_old_ddn_results_n101_k4_date_2023-12-11.p'
    filepath_results = 'results/NARMA-10_results_23/NARMA10_old_ddn_results_n51_k4_date_2023-12-09.p'
    try:
        with open(filepath_results, "rb") as f:
            print('Loading results from ' + filepath_results)
            es_dict = pkl.load(f)
    except:
        with open("../" + filepath_results, "rb") as f:
            print('Loading results from ' + filepath_results)
            es_dict = pkl.load(f)
    print(es_dict.keys())
    val_scores = es_dict['validation performance']
    max_gen = val_scores.shape[0]
    print(val_scores.shape)
    gens = np.arange(0, max_gen, 10)
    example_net = es_dict['example net']
    for gen in gens:
        print('generation =', gen)
        pop_val = val_scores[gen]
        pop_val_av = np.min(pop_val, axis=-1)
        pop_val_av = np.mean(pop_val_av, axis=-1)
        best_i = np.argmin(pop_val_av)
        params_best_i = es_dict['parameters'][gen][best_i]
        evaluate_networks(example_net, params_best_i, samples, data_train, data_val, results, gen)



    # Save overlap gridsearch
    save_name = "optimized_lag_sr_" + str(date.today()) + ".p"
    save_path = 'results/lag_analysis/' + save_name
    try:
        with open(save_path, "wb") as f:
            print('Saving results to ' + save_path)
            pkl.dump(results, f)
    except:
        with open("../" + save_path, "wb") as f:
            print('Saving results to ' + "../" + save_path)
            pkl.dump(results, f)