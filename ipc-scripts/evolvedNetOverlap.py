import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import numpy as np
from utils import eval_candidate_lag_gridsearch_NARMA, createNARMA30, region_specific_IPC
import pickle as pkl
from datetime import date

def get_best_scores(results_dict, max_best=True, lag_grid=False, agg=np.mean):
    val_scores = results_dict['validation performance']
    best_scores = []
    best_inds = []
    for i, pop_scores in enumerate(val_scores):
        if lag_grid:
            pop_scores = np.min(pop_scores, axis=-1)
        pop_scores = agg(pop_scores, axis=-1)
        if max_best:
            best_ind = np.argmax(pop_scores)
        else:
            best_ind = np.argmin(pop_scores)
        best_score = pop_scores[best_ind]
        best_scores.append(best_score)
        best_inds.append(best_ind)
    return best_scores, best_inds


def regional_IPC_evo(evo_dict, task_caps, train_data, test_data, alphas, reps_per_cand=5, warmup=400, max_gen=200,
                     interval=10, use_test_for_ipc=True):
    gen_list = np.arange(0, max_gen, interval)
    best_scores, best_inds = get_best_scores(evo_dict, False, True)
    ex_net = evo_dict['example net']

    performance_and_ipc = []

    if use_test_for_ipc:
        ipc_in = test_data[0, warmup:]
    else:
        ipc_in = train_data[0, warmup:]

    ipc_in = ipc_in * 4 - 1

    for gen in gen_list:
        print('Gen', gen)
        best_i = best_inds[gen]
        par = evo_dict['parameters'][gen, best_i]
        print('--- Evaluating candidate on test data and computing regional IPC')
        t_gen = []
        r_IPCs_gen = []
        for i in range(reps_per_cand):
            net = ex_net.get_new_network_from_serialized(par)
            _, test, _, train_states, val_states = eval_candidate_lag_gridsearch_NARMA(net, train_data, test_data,
                                                                                       warmup=warmup, alphas=alphas,
                                                                                       return_states=True)
            best_task_lag = np.argmin(test)
            print('--- Task lag: ' + str(best_task_lag))
            t_gen.append(test[best_task_lag])

            # evaluate IPC
            if use_test_for_ipc:
                ipc_states = val_states
            else:
                ipc_states = train_states

            ipc_in_clipped = ipc_in
            if best_task_lag > 0:
                ipc_in_clipped = ipc_in[:-best_task_lag]
            ipc_states_clipped = ipc_states[best_task_lag:]

            r_IPCs, total_measured_IPC = region_specific_IPC(ipc_states_clipped, task_caps, ipc_in_clipped)
            # overlap = estimate_from_IPC(task_caps, r_IPCs, degrees=[2])
            # overlap_gen.append(overlap)
            r_IPCs_gen.append(r_IPCs)
        print('val_score during evo: ' + str(best_scores[gen]))
        print('------ Test performance: ' + str(np.mean(t_gen)))
        # print('------ Overlap: ' + str(np.mean(overlap_gen)))

        performance_and_ipc.append((t_gen, r_IPCs_gen))
    return performance_and_ipc

def load_pkl_dict(path):
    try:
        with open(path, 'rb') as f:
            dict = pkl.load(f)
            return dict
    except FileNotFoundError:
        with open("../" + path, 'rb') as f:
            dict = pkl.load(f)
            return dict

def save_pkl_dict(path, dict):
    try:
        with open(path, 'wb') as f:
            pkl.dump(dict, f)
    except FileNotFoundError:
        with open("../" + path, 'wb') as f:
            pkl.dump(dict, f)

if __name__ == "__main__":
    NARMA_30_imports = {
        'DDN': {
            'path': "results/NARMA-30_results_23/NARMA30_old_ddn_results_n101_k4_date_2023-12-11.p"
        },
        'ESN': {
            'path': "results/NARMA-30_results_23/NARMA30_old_bl_results_n101_k4_date_2024-03-18.p"
        }
    }

    task_cap_path = 'results/ipc-results/narma_30_task_caps_2025-02-27.p'
    task_caps = load_pkl_dict(task_cap_path)
    task_caps = task_caps['taskCap']

    data_train = np.array(createNARMA30(5000)).reshape((2, 5000))
    data_test = np.array(createNARMA30(7000)).reshape((2, 7000))
    alphas = [10e-7, 10e-5, 10e-3]

    for net_type in NARMA_30_imports:
        filename = "narma_30_evolved_IPC_" + net_type + str(date.today()) + ".p"
        save_path = 'results/ipc-results/' + filename
        es_dict = load_pkl_dict(NARMA_30_imports[net_type]['path'])
        print("Computing IPC throughout evolution for " + net_type)
        ipc = regional_IPC_evo(es_dict, task_caps, data_train, data_test, alphas)
        results_dict ={
            'r_ipc': ipc,
            'task_caps': task_caps,
        }
        print("Saving evolution IPC as " + save_path)
        save_pkl_dict(save_path, results_dict)