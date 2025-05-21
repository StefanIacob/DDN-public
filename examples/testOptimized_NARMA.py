import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import numpy as np
import pickle as pkl
from utils import createNARMA10, createNARMA30, eval_candidate_lag_gridsearch_NARMA
from datetime import date
import argparse

EXPERIMENT_DATA = {
        'NARMA-30': {
            'DDN': {
                'path': "results/NARMA-30_results_23/NARMA30_old_ddn_results_n101_k4_date_2023-12-11.p"
            },
            'ESN': {
                'path': "results/NARMA-30_results_23/NARMA30_old_bl_results_n101_k4_date_2024-03-18.p"
            }
        },
        'NARMA-10': {
            'ESN': {
                'path': "results/NARMA-10_results_23/NARMA10_old_bl_results_n51_k4_date_2024-03-18.p"
            },
            'DDN': {
                'path': "results/NARMA-10_results_23/NARMA10_old_ddn_results_n51_k4_date_2023-12-09.p"
            }
        }
    }

def load_data(data_dict):
    for net_type in data_dict:
        path = data_dict[net_type]['path']
        try:
            with open(path, 'rb') as f:
                data = pkl.load(f)
                data_dict[net_type]['es_dict'] = data
        except:
            with open('../' + path, 'rb') as f:
                data = pkl.load(f)
                data_dict[net_type]['es_dict'] = data


def test_optimized(data_dict, resamples, data_train, data_test):
    results = {}
    for net_type in data_dict:
        print('Network type: {}'.format(net_type))
        print('Testing candidate for {} resamples...'.format(resamples))
        es_dict = data_dict[net_type]['es_dict']
        net = es_dict['example net']
        best_p = es_dict['evolutionary strategy'].best.x
        warmup = 400
        alphas = [10e-7, 10e-5, 10e-3]
        results[net_type] = []
        for resample in range(resamples):
            print('resample ' + str(resample) + '...')
            best_net = net.get_new_network_from_serialized(best_p)
            print('training readout and testing network...')
            t, v, m = eval_candidate_lag_gridsearch_NARMA(best_net, data_train, data_test, warmup=warmup, alphas=alphas)
            results[net_type].append(v)
            print('score: ' + str(np.min(v)))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n10", "--narma_10", action='store_true', help='Test N-10 experiment')
    parser.add_argument("-n30", "--narma_30", action='store_true', help='Test N-30 experiment')
    parser.add_argument("-r", "--repetitions", action='store', type=int, help='Nr of resamples per candidate')

    args = parser.parse_args()

    config = vars(args)
    test_n10 = config['narma_10']
    test_n30 = config['narma_30']
    network_resamples = config['repetitions']
    tasks = []
    if test_n10:
        np.random.seed(42)
        task_data = {
            'name': 'NARMA-10',
            'train': np.array(createNARMA10(20000)).reshape((2, 20000)),
            'test': np.array(createNARMA10(10000)).reshape((2, 10000))
        }
        np.random.seed()
        tasks.append(task_data)
    if test_n30:
        np.random.seed(42)
        task_data = {
            'name': 'NARMA-30',
            'train': np.array(createNARMA30(20000)).reshape((2, 20000)),
            'test': np.array(createNARMA30(10000)).reshape((2, 10000))
        }
        np.random.seed()
        tasks.append(task_data)
    if len(tasks) == 0:
        print('Neither N-10 nor N-30 selected, please add either -n10, -n30 or both flags to test any networks.')

    for task in tasks:
        task_name = task['name']
        data_train = task['train']
        data_test = task['test']
        print('Testing optimized ' + task_name + ' networks from ' + str(EXPERIMENT_DATA[task_name]))
        load_data(EXPERIMENT_DATA[task_name])
        task_results = test_optimized(EXPERIMENT_DATA[task_name], network_resamples, data_train, data_test)
        filename = 'test_' + task_name + '_' +  str(date.today()) + '.p'
        try:
            save_path = '../results/' + task_name + '_results_23/' + filename
            print('Testing ' + task_name + ' finished, saving test results as ' + save_path)
            with open(save_path, 'wb') as f:
                pkl.dump(task_results, f)
        except Exception as e:
            save_path = 'results/' + task_name + '_results_23/' + filename
            print('Failed, saving test results as ' + save_path)
            with open(save_path, 'wb') as f:
                pkl.dump(task_results, f)
