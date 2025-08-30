import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import numpy as np
from network import DistDelayNetwork, tanh_activation
from utils import eval_candidate_lag_gridsearch_NARMA, createNARMA30, region_specific_IPC, IPC_overlap
import pickle as pkl
from datetime import date
import argparse

def get_random_ddn(N, sr, a):
    """
    Generate a random ddn
    """
    connectivity = .5
    weights = np.random.uniform(-.5, .5, size=(N,N))
    for i in range(N):
        for j in range(N):
            if np.random.uniform() > connectivity:
                weights[i,j] = 0
    sr_current = np.max(np.absolute(+np.linalg.eigvals(weights)))
    weights = weights * (sr/sr_current)
    bias = np.random.uniform(0, 1, size=(N,))
    mu = [0,0]
    cov = [
            [.001, 0],
            [0, .001]
    ]

    coordinates = np.random.multivariate_normal(mu, cov, size=N)
    n_type = np.ones(shape=(N,))
    decay = a
    input_n = [0]
    coordinates[0, :] = [0,0]
    output_n = np.array(range(1, N))
    ddn = DistDelayNetwork(weights, bias, n_type, coordinates, decay, input_n, output_n,
                           activation_func=tanh_activation, dt=.0005)
    print('max D: ', np.max(ddn.D))
    return ddn


def best_lag_gridsearch(results, sr_grid, leak_grid, reps, data_train, data_val, alphas):
    sr_grid_size = len(sr_grid)
    leak_grid_size = len(leak_grid)
    total_reps = sr_grid_size * leak_grid_size * reps
    i = 0
    print('Total number of evaluations: ' + str(total_reps))
    for sr in sr_grid:
        for leak in leak_grid:
            for rep in range(reps):
                p_n_ddn = 10
                j = 0
                while p_n_ddn > 2:
                    # avoid outliers
                    if j > 0:
                        print('Outlier in measuring task performance, attempt', j)
                    j += 1
                    ddn_net = get_random_ddn(100, sr, leak)
                    # evaluate on NARMA
                    _, val_performance_per_lag_ddn, _, _, ddn_states = eval_candidate_lag_gridsearch_NARMA(ddn_net,
                                                                                                           data_train,
                                                                                                           data_val,
                                                                                                           400,
                                                                                                           alphas=alphas,
                                                                                                           return_states=True)
                    best_lag_ddn = np.argmin(val_performance_per_lag_ddn)
                    p_n_ddn = val_performance_per_lag_ddn[best_lag_ddn]

                print('ddn: ' + str(p_n_ddn) + ', lag: ' + str(best_lag_ddn))

                i += 1
                print(str((i / total_reps) * 100) + "% done")
                write_result(results, sr, leak, p_n_ddn,  best_lag_ddn)

def write_result(results, sr, leak, task_p, optimal_lag):
    results['sr'] = sr
    results['leak'] = leak
    results['task_p'] = task_p
    results['optimal_lag'] = optimal_lag


if __name__ == "__main__":
    data_train = np.array(createNARMA30(4000)).reshape((2, 4000))
    data_val = np.array(createNARMA30(6000)).reshape((2, 6000))
    ipc_in = data_val[0, 400:] * 4 - 1
    sr_grid_size = 12
    leak_grid_size = 12
    sr_grid = np.linspace(0.1, 2, sr_grid_size)
    leak_grid = np.linspace(0.3, 1, leak_grid_size)
    reps = 3
    alphas = [10e-7, 10e-5, 10e-3]

    results = {
        'sr': [],
        'leak': [],
        'task_p': [],
        'optimal_lag': [],
    }

    best_lag_gridsearch(results, sr_grid, leak_grid, reps, data_train, data_val, alphas)

    # Save overlap gridsearch
    save_name = "gridsearch_lag_sr_" + str(date.today()) + ".p"
    save_path = 'results/lag_analysis/' + save_name
    try:
        with open(save_path, "wb") as f:
            print('Saving results to ' + save_path)
            pkl.dump(results, f)
    except:
        with open("../" + save_path, "wb") as f:
            print('Saving results to ' + "../" + save_path)
            pkl.dump(results, f)