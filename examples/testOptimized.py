import numpy as np
from network import tanh_activation
import pickle as pkl
from utils import createNARMA30, eval_candidate_lag_gridsearch_NARMA
from datetime import date
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n10", "--narma_10", action="store_true", type=bool)
    parser.add_argument("-n30", "--narma_30", action="store_true", type=bool)

    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    experiment_data = {
        'NARMA-30': {
            'DDN': {
                'path': "NARMA-30_results_23/NARMA30_old_ddn_results_n101_k4_date_2023-12-11.p"
            },
            'ESN': {
                'path': "NARMA-30_results_23/NARMA30_old_bl_results_n101_k4_date_2024-03-18.p"
            }
        },
        'NARMA-10': {
            'ESN': {
                'path': "NARMA-10_results_23/NARMA10_old_bl_results_n51_k4_date_2024-03-18.p"
            },
            'DDN': {
                'path': "NARMA-10_results_23/NARMA10_old_ddn_results_n51_k4_date_2023-12-09.p"
            }
        }
    }

    np.random.seed(42)
    data_train_42 = np.array(createNARMA30(20000)).reshape((2, 20000))
    data_test_42 = np.array(createNARMA30(10000)).reshape((2, 10000))
    np.random.seed()

    network_resamples = 40

    test_results = {
        '1K': {
            'DDN': [],
            'ESN': []
        },
        '4K': {
            'DDN': [],
            'ESN': []
        }
    }

    # for k_size in experiment_data:
    #     for net_type in experiment_data[k_size]:
    #         print('Condition: ' + str(k_size) + ', ' + str(net_type))
    #         net = experiment_data[k_size][net_type]['example_net']
    #         best_p = experiment_data[k_size][net_type]['es_dict']['evolutionary strategy'].best.x
    #         for resample in range(network_resamples):
    #             print('resample ' + str(resample) + '...')
    #             best_net = net.get_new_network_from_serialized(best_p)
    #             print('training readout and testing network...')
    #             t, v, m = eval_candidate_lag_gridsearch(best_net, data_train_42, data_test_42, warmup=400,
    #                                                     alphas=alphas)
    #             test_results[k_size][net_type].append(v)
    #             print('score: ' + str(np.min(v)))
    # filename = 'test_' +  str(date.today()) + '.p'
    # save_file = 'results-2022-paper/test_results/' + filename
    # with open(save_file, 'wb') as f:
    #     pkl.dump(test_results, f)
