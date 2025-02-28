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
import Capacities.capacities as CAP

def get_random_ddn(N, sr, a):
    """
    Generate a random ddn and esn, which are (except for delays) identical to each other, with a specified spectral
    radius and leak rate.
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
    esn = DistDelayNetwork(weights, bias, n_type, coordinates, decay, input_n, output_n,
                           activation_func=tanh_activation, dt=1)
    return ddn, esn

def write_result(results, sr, leak, task_p, regional_cap, overlap, net_type):
    results['sr'].append(sr)
    results['leak'].append(leak)
    results['task_p'].append(task_p)
    results['regional_cap'].append(regional_cap)
    results['overlap'].append(overlap)
    results['net_type'].append(net_type)


def full_IPC(inputs, states, maxdel=35, maxdeg=2, maxvars=2):
    Citer=CAP.capacity_iterator(mindel=1,mindeg=1, maxdeg=maxdeg, minvars=1,maxvars=maxvars, maxdel=maxdel, delskip=100,
                            m_delay=False,
                            m_windowpos=False, m_window=False, m_powerlist=False,m_variables=False,
                            m_degrees=False, minwindow=0, maxwindow=31)
    totalcap, allcaps, numcaps, nodes = Citer.collect(inputs, states)
    return allcaps


def overlap_gridsearch(results, sr_grid, leak_grid, reps, task_caps, full_ipc=False):
    total_reps = sr_grid_size * leak_grid_size * reps
    i = 0
    print('Total number of evaluations: ' + str(total_reps))
    for sr in sr_grid:
        for leak in leak_grid:
            for rep in range(reps):
                p_n_ddn = 10
                p_n_bl = 10
                j = 0
                while p_n_ddn > 2 or p_n_bl > 2:
                    # avoid outliers
                    print('Measuring task performance', j)
                    j += 1
                    ddn_net, bl_net = get_random_ddn(100, sr, leak)
                    # evaluate on NARMA
                    _, val_performance_per_lag_ddn, _, _, ddn_states = eval_candidate_lag_gridsearch_NARMA(ddn_net,
                                                                                                           data_train,
                                                                                                           data_val,
                                                                                                           400,
                                                                                                           alphas=alphas,
                                                                                                           return_states=True)
                    best_lag_ddn = np.argmin(val_performance_per_lag_ddn)
                    p_n_ddn = val_performance_per_lag_ddn[best_lag_ddn]

                    _, val_performance_per_lag_bl, _, _, bl_states = eval_candidate_lag_gridsearch_NARMA(bl_net,
                                                                                                         data_train,
                                                                                                         data_val, 400,
                                                                                                         alphas=alphas,
                                                                                                         return_states=True)
                    best_lag_bl = np.argmin(val_performance_per_lag_bl)
                    p_n_bl = val_performance_per_lag_bl[best_lag_bl]

                ipc_in_clipped_ddn = ipc_in
                if best_lag_ddn > 0:
                    ipc_in_clipped_ddn = ipc_in[:-best_lag_ddn]
                ddn_states_clipped = ddn_states[best_lag_ddn:]

                ipc_in_clipped_bl = ipc_in
                if best_lag_bl > 0:
                    ipc_in_clipped_bl = ipc_in[:-best_lag_bl]
                bl_states_clipped = bl_states[best_lag_bl:]

                print('ddn: ' + str(p_n_ddn) + ', lag: ' + str(best_lag_ddn))
                print('bl: ' + str(p_n_bl) + ', lag: ' + str(best_lag_bl))

                # evaluate IPC
                if full_ipc:
                    # Parameters the same as in paper
                    r_IPCs_ddn = full_IPC(ipc_in_clipped_ddn, ddn_states_clipped, 60, 3)
                    r_IPCs_bl = full_IPC(ipc_in_clipped_bl, bl_states_clipped, 60, 3)
                else:
                    r_IPCs_ddn, _ = region_specific_IPC(ddn_states_clipped, task_caps, ipc_in_clipped_ddn)
                    r_IPCs_bl, _ = region_specific_IPC(bl_states_clipped, task_caps, ipc_in_clipped_bl)

                # estimate overlap
                overlap_ddn = IPC_overlap(task_caps, r_IPCs_ddn)
                overlap_bl = IPC_overlap(task_caps, r_IPCs_bl)

                i += 1
                print(str((i / total_reps) * 100) + "% done")
                write_result(results, sr, leak, p_n_bl, r_IPCs_bl, overlap_bl, 'ESN')
                write_result(results, sr, leak, p_n_ddn, r_IPCs_ddn, overlap_ddn, 'DDN')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--full_ipc", action='store_true', help='compute full IPC '
                                                                      'instead of regional ipc')

    args = parser.parse_args()
    config = vars(args)
    full_ipc = config['full_ipc']

    data_train = np.array(createNARMA30(4000)).reshape((2, 4000))
    data_val = np.array(createNARMA30(6000)).reshape((2, 6000))
    ipc_in = data_val[0, 400:] * 4 - 1
    sr_grid_size = 12
    leak_grid_size = 12
    sr_grid = np.linspace(0.1, 2, sr_grid_size)
    leak_grid = np.linspace(0.3, 1, leak_grid_size)
    reps = 3
    alphas = [10e-7, 10e-5, 10e-3]

    # Load task caps
    tc_path = "results/ipc-results/narma_30_task_caps_2025-02-27.p"
    print("Loading task capacity from " + tc_path)
    try:
        with open(tc_path, "rb") as f:
            tc_dict = pkl.load(f)
    except:
        with open("../" + tc_path, "rb") as f:
            tc_dict = pkl.load(f)
    task_allcaps_30 = tc_dict['taskCap']
    print("Spectral radius grid: " + str(sr_grid))
    print("Leak rate grid " + str(leak_grid))

    results = {
        'sr': [],
        'leak': [],
        'task_p': [],
        'regional_cap': [],
        'overlap': [],
        'net_type': []
    }

    overlap_gridsearch(results, sr_grid, leak_grid, reps, task_allcaps_30, full_ipc=full_ipc)

    # Save overlap gridsearch
    save_name = "gridsearch_ipc_overlap_" + str(date.today()) + ".p"
    save_path = 'results/ipc-results/' + save_name
    try:
        with open(save_path, "wb") as f:
            print('Saving results to ' + save_path)
            pkl.dump(results, f)
    except:
        with open("../" + save_path, "wb") as f:
            print('Saving results to ' + "../" + save_path)
            pkl.dump(results, f)