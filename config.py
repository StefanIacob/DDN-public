import numpy as np

propagation_vel = 30 # m/s

def get_p_dict_heterogeneity_exp(K, x_range, y_range, start_location_var=0.003, start_location_mean_var=0,
                                 start_weight_mean=.2, start_weight_var=.3, start_bias_mean=0, start_bias_var=.5):
    # returns base p_dict, requires further tweaking for different experiment conditions
    center = 0
    start = center-start_location_mean_var
    end = center+start_location_mean_var
    step = (end-start)/K
    start_locations_on_diag = [start + (i+1) * step for i in range(K)]
    p_dict = {
        'mix': {
            'val': np.ones(shape=(K,)) / K,
            'evolve': True,
            'range': (0, 1),
            'lims': (0, float('inf'))
        },
        'mu_x': {
            'val': np.array(start_locations_on_diag),
            'evolve': True,
            'range': x_range,
            'lims': x_range
        },
        'mu_y': {
            'val': np.array(start_locations_on_diag),
            'evolve': True,
            'range': y_range,
            'lims': y_range
        },
        'variance_x': {
            'val': np.ones(shape=(K,)) * start_location_var,
            'evolve': True,
            'range': x_range,
            'lims': (0, float('inf'))
        },
        'variance_y': {
            'val': np.ones(shape=(K,)) * start_location_var,
            'evolve': True,
            'range': y_range,
            'lims': (0, float('inf'))
        },
        'correlation': {
            'val': np.ones(shape=(K,)) * 0,
            'evolve': True,
            'range': (-1, 1),
            'lims': (-1, 1)
        },
        'inhibitory': {
            'val': np.ones(shape=(K,)) * 0.4,
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        },
        'connectivity': {
            'val': np.ones(shape=(K, K)) * 0.1,
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        },
        'weight_mean': {
            'val': np.ones(shape=(K, K)) * start_weight_mean,
            'evolve': True,
            'range': (0, 2),
            'lims': (0, float('inf'))
        },
        'weight_scaling': {
            'val': np.ones(shape=(K, K)) * start_weight_var,
            'evolve': True,
            'range': (0, 2),
            'lims': (0, float('inf'))
        },
        'bias_mean': {
            'val': np.ones(shape=(K,)) * start_bias_mean,
            'evolve': True,
            'range': (-1, 1),
            'lims': (-float('inf'), float('inf'))
        },
        'bias_scaling': {
            'val': np.ones(shape=(K,)) * start_bias_var,
            'evolve': True,
            'range': (0, 1),
            'lims': (0, float('inf'))
        },
        'decay_mean': {
            'val': np.array([0.9]),
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        },
        'decay_scaling': {
            'val': np.array([.1]),
            'evolve': True,
            'range': (0, 1),
            'lims': (0, float('inf'))
        },
        'in_mean': {
            'val': .8 * np.ones(shape=(K,)),
            'evolve': True,
            'range': (-2, 2),
            'lims': (-float('inf'), float('inf'))
        },
        'in_scaling': {
            'val': .1 * np.ones(shape=(K,)),
            'evolve': True,
            'range': (0, 2),
            'lims': (0, float('inf'))
        },
        'in_connectivity': {
            'val': .6 * np.ones(shape=(K,)),
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        }
    }
    return p_dict