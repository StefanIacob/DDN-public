import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import numpy as np
from network import DistDelayNetwork, tanh_activation
from utils import eval_candidate_lag_gridsearch_NARMA, createNARMA30, region_specific_IPC, IPC_overlap, full_IPC
import pickle as pkl
from datetime import date
from simulator import NetworkSimulator



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

    narma_in, narma_out = createNARMA30(10000)
    maxdeg = 2
    maxdel = 45
    maxvar = 2
    reps = 10
    maxwin = 31

    for net_type in NARMA_30_imports:
        print("Computing IPC for " + net_type)
        filename = "best_narma_30_evolved_full_IPC_" + net_type + str(date.today()) + ".p"
        save_path = 'results/ipc-results/' + filename
        es_dict = load_pkl_dict(NARMA_30_imports[net_type]['path'])
        net = es_dict['example net']
        best_p = es_dict['evolutionary strategy'].best.x
        full_caps_list = []
        print('Resampling network and computing IPC ' + str(reps) + " times")
        for rep in range(reps):
            print("rep: " + str(rep) + ". Generating best network...")
            best_net = net.get_new_network_from_serialized(best_p)
            print("Computing IPC...")
            all_caps = full_IPC(best_net, narma_in, maxdel=maxdel, maxdeg=maxdeg, maxvars=maxvar, maxwin=maxwin)
            results_dict = {
                'ipc': all_caps,
                'maxdel': maxdel,
                'maxdeg': maxdeg,
                'maxwin': maxwin
            }
            full_caps_list.append(results_dict)

        print("Saving best network IPC as " + save_path)
        save_pkl_dict(save_path, full_caps_list)