propagation_vel = 30
hyperparameter_ranges = {
    # CMA-ES assumes initial parameters are centered around 0 and have the same pre-defined starting std.
    # The ranges below indicate to what range these CMA-ES Parameters need to be scaled for use in DDNs.
    'mu_x': [-1, 1],
    'mu_y': [-1, 1],
    'variance_x': [0, 1],
    'variance_y': [0, 1],
    'mix': [0, 1],
    'correlation': [-1, 1],
    'connectivity': [0, 1],
    'cluster_decay': [0, 1],
    'bias_scaling': [0, 2],
    'bias_mean': [-1, 1],
    'weight_scaling': [0, 2],
    'weight_mean': [-1, 1],
    'lr_scaling': [0, 1],
    'lr_mean': [0, 1],
    'y0_scaling': [0, 2],
    'y0_mean': [0.25, 1]
}
fixed_hyperparameters = {
    'mu_x': False,
    'mu_y': False,
    'variance_x': False,
    'variance_y': False,
    'mix': False,
    'correlation': False,
    'connectivity': False,
    'cluster_decay': False,
    'bias_scaling': False,
    'bias_mean': False,
    'weight_scaling': False,
    'weight_mean': False,
    'lr_scaling': False,
    'lr_mean': False,
    'y0_scaling': False,
    'y0_mean': False
}
