import numpy as np
from populations import get_empty_GMMpop
from network import tanh_activation
import pickle as pkl
from utils import createNARMA10, eval_candidate_lag_gridsearch
from datetime import date

if __name__ == '__main__':
    # Parameters regarding neuron location and activation function that was not stored in the pickle
    x_range = (0, .002)
    y_range = (0, .004)
    activation_function = tanh_activation
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    in_loc = (x_range[0] + width * 0.3, y_range[0] + height * 0.2)
    N = 301

    experiment_data = {
        '1K': {
            'DDN':
                {
                    'path': 'results-2022-paper/cma_es_gmm_200222k1_tanh_vd_reg.p',
                    'N': N,
                    'K': 1,
                    'B': 20,
                    'dt': .000008,
                    'x_range': x_range,
                    'y_range': y_range,
                    'in_loc': in_loc,
                    'activation_function': activation_function
                },
            'ESN':
                {
                    'path': 'results-2022-paper/cma_es_gmm_250322_k1_tanh_final_paper_bl.p',
                    'N': N,
                    'K': 1,
                    'B': 1,
                    'dt': .5,
                    'x_range': x_range,
                    'y_range': y_range,
                    'in_loc': in_loc,
                    'activation_function': activation_function
                }
        },
        '4K': {
            'DDN':
                {
                    'path': 'results-2022-paper/cma_es_gmm_230222k4_tanh_vd_reg.p',
                    'N': N,
                    'K': 4,
                    'B': 20,
                    'dt': .000008,
                    'x_range': x_range,
                    'y_range': y_range,
                    'in_loc': in_loc,
                    'activation_function': activation_function
                },
            'ESN':
                {
                    'path': 'results-2022-paper/cma_es_gmm_230222k4_tanh_bl_reg.p',
                    'N': N,
                    'K': 4,
                    'B': 1,
                    'dt': .5,
                    'x_range': x_range,
                    'y_range': y_range,
                    'in_loc': in_loc,
                    'activation_function': activation_function
                }
        }
    }

    for k_size in experiment_data:
        for net_type in experiment_data[k_size]:
            path = experiment_data[k_size][net_type]['path']
            with open(path, 'rb') as f:
                es_dict = pkl.load(f)
            print(es_dict.keys())
            experiment_data[k_size][net_type]['es_dict'] = es_dict

            example_net_parameters = {
                'N': experiment_data[k_size][net_type]['N'],
                'k': experiment_data[k_size][net_type]['K'],
                'dt': experiment_data[k_size][net_type]['dt'],
                'x_range': experiment_data[k_size][net_type]['x_range'],
                'y_range': experiment_data[k_size][net_type]['y_range'],
                'in_loc': experiment_data[k_size][net_type]['in_loc'],
                'activation_function': experiment_data[k_size][net_type]['activation_function'],
                'insize': 1,
                'buffersize': experiment_data[k_size][net_type]['B']
            }

            example_net = get_empty_GMMpop(**example_net_parameters)
            experiment_data[k_size][net_type]['example_net'] = example_net

    np.random.seed(42)
    data_train_42 = np.array(createNARMA10(20000)).reshape((2, 20000))
    data_test_42 = np.array(createNARMA10(10000)).reshape((2, 10000))
    np.random.seed()
    alphas = [10e-9, 10e-8, 10e-7]

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

    for k_size in experiment_data:
        for net_type in experiment_data[k_size]:
            print('Condition: ' + str(k_size) + ', ' + str(net_type))
            net = experiment_data[k_size][net_type]['example_net']
            best_p = experiment_data[k_size][net_type]['es_dict']['evolutionary strategy'].best.x
            for resample in range(network_resamples):
                print('resample ' + str(resample) + '...')
                best_net = net.get_new_network_from_serialized(best_p)
                print('training readout and testing network...')
                t, v, m = eval_candidate_lag_gridsearch(best_net, data_train_42, data_test_42, warmup=400,
                                                        alphas=alphas)
                test_results[k_size][net_type].append(v)
                print('score: ' + str(np.min(v)))
    filename = 'test_' +  str(date.today()) + '.p'
    save_file = 'results-2022-paper/test_results/' + filename
    with open(save_file, 'wb') as f:
        pkl.dump(test_results, f)
