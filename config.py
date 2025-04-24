import numpy as np

propagation_vel = 30  # m/s


def get_p_dict_like_p3_NARMA(K, x_range, y_range, start_location_var=0.003, start_location_mean_var=0):
    """
    Generates a dictionary of DDN hyperparameters with corresponding evolution rules, similar to the configuration of
    the experiment presented in https://github.com/StefanIacob/DDN-public/, for the NARMA exeperiments.

    Returns: parameter dictionary
    """
    center = 0
    start = center - start_location_mean_var
    end = center + start_location_mean_var
    step = (end - start) / K
    start_locations_on_diag = [start + (i + 1) * step for i in range(K)]
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
            'val': np.ones(shape=(K,)) * 0,
            'evolve': False,
            'range': (0, 1),
            'lims': (0, 1)
        },
        'connectivity': {
            'val': np.ones(shape=(K, K)) * 0.9,
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        },
        'weight_mean': {
            'val': np.ones(shape=(K, K)) * 0,
            'evolve': False,
            'range': (0, 2),
            'lims': (-float('inf'), float('inf'))
        },
        'weight_scaling': {
            'val': np.ones(shape=(K, K)) * 0.6,
            'evolve': True,
            'range': (0, 2),
            'lims': (0, float('inf'))
        },
        'bias_mean': {
            'val': np.ones(shape=(K,)) * 0,
            'evolve': False,
            'range': (-1, 1),
            'lims': (-float('inf'), float('inf'))
        },
        'bias_scaling': {
            'val': np.ones(shape=(K,)) * 0.5,
            'evolve': True,
            'range': (0, 2),
            'lims': (0, float('inf'))
        },
        'decay_mean': {
            'val': np.ones(shape=(K,)) * 0.95,
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        },
        'decay_scaling': {
            'val': np.ones(shape=(K,)) * 0,
            'evolve': False,
            'range': (0, 1),
            'lims': (0, float('inf'))
        },
        'in_mean': {
            'val': 0 * np.ones(shape=(K,)),
            'evolve': False,
            'range': (-2, 2),
            'lims': (-float('inf'), float('inf'))
        },
        'in_scaling': {
            'val': .9 * np.ones(shape=(K,)),
            'evolve': True,
            'range': (0, 2),
            'lims': (0, float('inf'))
        },
        'in_connectivity': {
            'val': .9 * np.ones(shape=(K,)),
            'evolve': True,
            'range': (0, 1),
            'lims': (0, 1)
        }
    }
    return p_dict

def get_p_dict_like_p3_MG(K, x_range, y_range):
    """
    Generates a dictionary of DDN hyperparameters with corresponding evolution rules, similar to the configuration of
    the experiment presented in https://github.com/StefanIacob/DDN-public/, for the NARMA exeperiments.

    Returns: parameter dictionary, should contain the following keys:
        'mix', 'mu_x', 'mu_y', 'variance_x', 'variance_y', 'correlation', 'inhibitory',
        'connectivity', 'weight_mean', 'weight_scaling', 'bias_mean', 'bias_scaling',
        'decay_mean', 'decay_scaling', 'in_scaling', 'in_mean', 'in_connectivity', 'lr_mean',
        'lr_scaling', 'theta0_mean', 'theta0_scaling', 'in_lr_mean', 'in_lr_scaling'
    """
    width = x_range[1] - x_range[0]
    start_location_var = width * 0.1
    start_location_mean_var = width/2.5
    center = 0
    start = center - start_location_mean_var
    end = center + start_location_mean_var
    step = (end - start) / K
    start_locations_on_diag = [start + (i + 1) * step for i in range(K)]
    p_dict = {'mix': {
        'val': np.ones(shape=(K,)) / K,
        'evolve': True,
        'range': (0, 1),
        'lims': (0, float('inf'))
    }, 'mu_x': {
        'val': np.array(start_locations_on_diag),
        'evolve': True,
        'range': x_range,
        'lims': x_range
    }, 'mu_y': {
        'val': np.array(start_locations_on_diag),
        'evolve': True,
        'range': y_range,
        'lims': y_range
    }, 'variance_x': {
        'val': np.ones(shape=(K,)) * start_location_var,
        'evolve': True,
        'range': [0, 1],
        'lims': (0, float('inf'))
    }, 'variance_y': {
        'val': np.ones(shape=(K,)) * start_location_var,
        'evolve': True,
        'range': [0, 1],
        'lims': (0, float('inf'))
    }, 'correlation': {
        'val': np.ones(shape=(K,)) * 0,
        'evolve': True,
        'range': (-1, 1),
        'lims': (-1, 1)
    }, 'inhibitory': {
        'val': np.ones(shape=(K,)) * 0,
        'evolve': False,
        'range': (0, 1),
        'lims': (0, 1)
    }, 'connectivity': {
        'val': np.ones(shape=(K, K)) * 0.9,
        'evolve': True,
        'range': (0, 1),
        'lims': (0, 1)
    }, 'weight_mean': {
        'val': np.ones(shape=(K, K)) * 0,
        'evolve': True,
        'range': (-1, 1),
        'lims': (-float('inf'), float('inf'))
    }, 'weight_scaling': {
        'val': np.ones(shape=(K, K)) * .5,
        'evolve': True,
        'range': (0, 2),
        'lims': (0, float('inf'))
    }, 'bias_mean': {
        'val': np.ones(shape=(K,)) * 0,
        'evolve': True,
        'range': (-1, 1),
        'lims': (-float('inf'), float('inf'))
    }, 'bias_scaling': {
        'val': np.ones(shape=(K,)) * .5,
        'evolve': True,
        'range': (0, 2),
        'lims': (0, float('inf'))
    }, 'decay_mean': {
        'val': np.ones(shape=(K,)) * 0.99,
        'evolve': True,
        'range': (0, 1),
        'lims': (0, 1)
    }, 'decay_scaling': {
        'val': np.ones(shape=(K,)) * 0,
        'evolve': False,
        'range': (0, 1),
        'lims': (0, float('inf'))
    }, 'in_mean': {
        'val': .0 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (-1, 1),
        'lims': (-float('inf'), float('inf'))
    }, 'in_scaling': {
        'val': .5 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (0, 2),
        'lims': (0, float('inf'))
    }, 'in_connectivity': {
        'val': .9 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (0, 1),
        'lims': (0, 1)
    }, 'lr_mean': {
        'val': .01 * np.ones(shape=(K, K)),
        'evolve': True,
        'range': (0, 1),
        'lims': (0, float('inf'))
    }, 'lr_scaling': {
        'val': .01 * np.ones(shape=(K, K)),
        'evolve': True,
        'range': (0, 1),
        'lims': (0, float('inf'))
    }, 'theta0_mean': {
        'val': .8 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (.25, 1),
        'lims': (0.25, float('inf'))  # TODO: why?
    }, 'theta0_scaling': {
        'val': .1 * np.ones(shape=(K)),
        'evolve': True,
        'range': (0, 2),
        'lims': (0, float('inf'))
    }, 'in_lr_mean': {
        'val': .01 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (0, 1),
        'lims': (0, float('inf'))
    }, 'in_lr_scaling': {
        'val': .01 * np.ones(shape=(K)),
        'evolve': True,
        'range': (0, 1),
        'lims': (0, float('inf'))
    }}

    return p_dict

def get_p_dict_heterogeneity_exp(K, x_range, y_range, start_location_var=0.003, start_location_mean_var=0,
                                 start_weight_mean=.2, start_weight_var=.3, start_bias_mean=0, start_bias_var=.5):
    # returns base p_dict, requires further tweaking for different experiment conditions
    center = 0
    start = center - start_location_mean_var
    end = center + start_location_mean_var
    step = (end - start) / K
    start_locations_on_diag = [start + (i + 1) * step for i in range(K)]
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


def get_p_dict_heterogeneity_exp_adaptive(K, x_range, y_range, start_location_var=0.003, start_location_mean_var=0,
                                          start_weight_mean=.2, start_weight_var=.3, start_bias_mean=0,
                                          start_bias_var=.5):
    # returns base p_dict, requires further tweaking for different experiment conditions
    p_dict = get_p_dict_heterogeneity_exp(K, x_range, y_range, start_location_var=start_location_var,
                                          start_location_mean_var=start_location_mean_var,
                                          start_weight_mean=start_weight_mean, start_weight_var=start_weight_var,
                                          start_bias_mean=start_bias_mean, start_bias_var=start_bias_var)

    p_dict['lr_mean'] = {
        'val': .1 * np.ones(shape=(K, K)),
        'evolve': True,
        'range': (0, .3),
        'lims': (0, float('inf'))
    }
    p_dict['lr_scaling'] = {
        'val': .1 * np.ones(shape=(K, K)),
        'evolve': True,
        'range': (0, .1),
        'lims': (0, float('inf'))
    }
    p_dict['theta0_mean'] = {
        'val': .1 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (0, .25),
        'lims': (0, .25)  # TODO: why?
    }
    p_dict['theta0_scaling'] = {
        'val': .1 * np.ones(shape=(K)),
        'evolve': True,
        'range': (0, .1),
        'lims': (0, float('inf'))
    }
    p_dict['in_lr_mean'] = {
        'val': .1 * np.ones(shape=(K,)),
        'evolve': True,
        'range': (0, .3),
        'lims': (0, float('inf'))
    }
    p_dict['in_lr_scaling'] = {
        'val': .1 * np.ones(shape=(K)),
        'evolve': True,
        'range': (0, .1),
        'lims': (0, float('inf'))
    }
    # p_dict['out_lr_mean'] = p_dict['in_lr_mean']
    # p_dict['out_lr_scaling'] = p_dict['in_lr_scaling']
    # p_dict['out_theta0'] = {
    #     'val': np.array([.1]),
    #     'evolve': True,
    #     'range': (0, .25),
    #     'lims': (0, .25)
    # }
    # p_dict['out_mean'] = {
    #     'val': .8 * np.ones(shape=(K,)),
    #     'evolve': True,
    #     'range': (-2, 2),
    #     'lims': (-float('inf'), float('inf'))
    # }
    # p_dict['out_scaling'] = {
    #     'val': .1 * np.ones(shape=(K,)),
    #     'evolve': True,
    #     'range': (0, 2),
    #     'lims': (0, float('inf'))
    # }
    # p_dict['out_connectivity'] = {
    #     'val': .6 * np.ones(shape=(K,)),
    #     'evolve': True,
    #     'range': (0, 1),
    #     'lims': (0, 1)
    # }

    return p_dict
