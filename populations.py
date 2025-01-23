import numpy as np
from network import DistDelayNetwork, tanh_activation, sigmoid_activation
from config import hyperparameter_ranges, fixed_hyperparameters

class GMMPopulation(DistDelayNetwork):
    """
    Class for a rate-based continuous neuron population that implements time delays based on signal propagation
    distance. Neurons are positioned in clusters, with coordinates sampled from a gaussian mixture.
    """

    def __init__(self, N, mix, mu, variance, correlation, inhibitory, connectivity, cluster_connectivity,
                 weight_scaling, weight_mean, bias_scaling, bias_mean, decay, size_in, size_out, in_loc,
                 act_func=tanh_activation, autapse=False, dt=0.0001, x_lim=None,
                 y_lim=None, fixed_delay=0):

        assert (x_lim is None) == (y_lim is None), "For now it is not possible to have only one of the two axes limited"

        self.size_in = size_in
        self.size_out = size_out
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.k = len(mix)
        self.mix = mix
        self.mu_x = mu[:, 0]
        self.mu_y = mu[:, 1]
        self.variance_x = variance[:, 0]
        self.variance_y = variance[:, 1]
        self.correlation = correlation
        self.inhibitory = inhibitory
        self.connectivity = connectivity
        self.cluster_connectivity = cluster_connectivity
        self.weight_scaling = weight_scaling
        self.weight_mean = weight_mean
        self.bias_scaling = bias_scaling
        self.bias_mean = bias_mean
        self.in_loc = in_loc
        self.autapse=autapse
        self.cluster_decay = decay
        self.fixed_delay=fixed_delay

        if x_lim is None:
            start_width = 2
            start_height = 2
            r_mu_x = [-1, 1]
            r_mu_y = [-1, 1]

        else:
            start_width = x_lim[1] - x_lim[0]
            start_height = y_lim[1] - y_lim[0]
            r_mu_x = [x_lim[0], x_lim[1]]
            r_mu_y = [y_lim[0], y_lim[1]]

        r_variance_x = [0, start_width/2]
        r_variance_y = [0, start_height/2]

        r_mix = [0, 1]
        r_corr = [-1, 1]
        r_inh = [0, 1]
        r_conn = [0, 1]
        r_decay = [0, 1]
        r_bias_scaling = [0, 2]
        r_bias_mean = [-1, 1]
        r_weight_scaling = [0, 2]
        r_weight_mean = [-1, 1]

        self.ranges = [r_mix, r_mu_x, r_mu_y, r_variance_x, r_variance_y, r_corr, r_conn, r_weight_scaling,
                       r_weight_mean, r_bias_scaling, r_bias_mean, r_decay]
        self.centers = [(r[1] + r[0]) / 2 for r in self.ranges]
        self.scales = [r[1] - r[0] for r in self.ranges]

        k = self.k
        var_mat = np.zeros((k, 2, 2))
        corr_mat = np.ones((k, 2, 2))
        for i in range(k):
            var_mat[i, 0, 0] = self.variance_x[i]
            var_mat[i, 1, 1] = self.variance_y[i]
            corr_mat[i, 0, 1] = self.correlation[i]
            corr_mat[i, 1, 0] = self.correlation[i]

        Sigma = np.matmul(np.matmul(var_mat, corr_mat), var_mat)
        self.covariances = Sigma
        grid, n_type, clusters = get_gaussian_mixture_config(N - size_in, mix, mu, inhibitory, Sigma,
                                                             x_range=x_lim, y_range=y_lim)

        # width = x_range[1] - x_range[0]
        # height = y_range[1] - y_range[0]

        mu_x_in, mu_y_in = in_loc

        Sigma_in = 0.002 * (np.array(
            [
                [start_width, 0],
                [0, start_height]
            ])) ** 2
        Sigma_in = np.expand_dims(Sigma_in, 0)

        grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, size_in, mu_x_in, mu_y_in,
                                                   Sigma=Sigma_in,
                                                   x_range=x_lim, y_range=y_lim)
        input_index = np.array(range(0, size_in))
        output_index = np.random.choice(range(size_in, N), size=(size_out, 1), replace=False)

        bias = np.zeros((N,))
        n_decay = np.zeros((N,))
        for i in range(N):
            c = int(clusters[i])
            sf = bias_scaling[c]
            bias[i] = np.random.uniform(-0.5, 0.5) * sf + bias_mean[c]
            n_decay[i] = decay[c]

        W = get_weight_matrix(weight_mean, weight_scaling, clusters)
        if not autapse:
            np.fill_diagonal(W, 0)  # No self connections
        W = clustered_pop_pruning(W, connectivity, clusters, cluster_connectivity)

        super().__init__(weights=W, bias=bias, n_type=n_type, coordinates=grid, decay=n_decay, input_n=input_index,
                         output_n=output_index, activation_func=act_func, dt=dt)
        if fixed_delay > 0:
            self.D = np.ones_like(self.D) * fixed_delay

    def get_parameters_from_serialized(self, serialized_parameters):
        serialized_parameters = np.array(serialized_parameters)

        def scale_up(params, middle, scale):
            scaled = params * scale
            scaled = scaled + middle
            return scaled

        # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, weight_mean, bias_scaling, bias_mean,
        # decay

        # mix: Apply limit at 0 and normalize to get proper distribution
        mix = scale_up(np.array(serialized_parameters[:self.k]), self.centers[0], self.scales[0])
        serialized_parameters = serialized_parameters[self.k:]
        mix = np.maximum(0, mix)
        sum_mix = np.sum(mix)
        if sum_mix != 0:
            mix = mix / np.sum(mix)
        else:
            mix = np.ones_like(mix) / self.k

        # mu_x: Apply hard limit to not exceed ranges
        mu_x = scale_up(np.array(serialized_parameters[:self.k]), self.centers[1], self.scales[1])
        serialized_parameters = serialized_parameters[self.k:]
        mu_x = np.maximum(mu_x, self.x_range[0])
        mu_x = np.minimum(mu_x, self.x_range[1])

        # mu_y: Apply hard limit to not exceed ranges
        mu_y = scale_up(np.array(serialized_parameters[:self.k]), self.centers[2], self.scales[2])
        serialized_parameters = serialized_parameters[self.k:]
        mu_y = np.maximum(mu_y, self.y_range[0])
        mu_y = np.minimum(mu_y, self.y_range[1])

        mu = np.zeros((self.k, 2))
        mu[:, 0] = mu_x
        mu[:, 1] = mu_y

        # var_x: Make sure var is non-negative)
        var_x = scale_up(np.array(serialized_parameters[:self.k]), self.centers[3], self.scales[3])
        serialized_parameters = serialized_parameters[self.k:]
        var_x = np.maximum(0, var_x)

        # var_y: Make sure var is non-negative)
        var_y = scale_up(np.array(serialized_parameters[:self.k]), self.centers[4], self.scales[4])
        serialized_parameters = serialized_parameters[self.k:]
        var_y = np.maximum(0, var_y)

        var = np.zeros((self.k, 2))
        var[:, 0] = var_x
        var[:, 1] = var_y

        # corr_xy: Make sure correlation is between -1 and 1
        correlation = scale_up(np.array(serialized_parameters[:self.k]), self.centers[5], self.scales[5])
        serialized_parameters = serialized_parameters[self.k:]
        correlation = np.minimum(np.maximum(correlation, -0.99), 0.99)

        # conn: Make sure connectivity is between 0 and 1
        connectivity = scale_up(np.array(serialized_parameters[:(self.k + 1) ** 2]), self.centers[6], self.scales[6])
        connectivity = np.reshape(connectivity, (self.k + 1, self.k + 1))
        serialized_parameters = serialized_parameters[(self.k + 1) ** 2:]
        connectivity = np.minimum(np.maximum(connectivity, 0), 1)

        # weight_scaling: Weight scaling should be non-negative
        weight_scaling = scale_up(np.array(serialized_parameters[:(self.k + 1) ** 2]), self.centers[7], self.scales[7])
        weight_scaling = np.reshape(weight_scaling, (self.k + 1, self.k + 1))
        serialized_parameters = serialized_parameters[(self.k + 1) ** 2:]
        weight_scaling = np.maximum(weight_scaling, 0)

        # weight_mean
        weight_mean = scale_up(np.array(serialized_parameters[:(self.k + 1) ** 2]), self.centers[8], self.scales[8])
        weight_mean = np.reshape(weight_mean, (self.k + 1, self.k + 1))
        serialized_parameters = serialized_parameters[(self.k + 1) ** 2:]

        # bias_scaling: Bias scaling should be non-negative
        bias_scaling = scale_up(np.array(serialized_parameters[:self.k + 1]), self.centers[9], self.scales[9])
        serialized_parameters = serialized_parameters[self.k + 1:]
        bias_scaling = np.maximum(bias_scaling, 0)

        # bias_mean
        bias_mean = scale_up(np.array(serialized_parameters[:self.k + 1]), self.centers[10], self.scales[10])
        serialized_parameters = serialized_parameters[self.k + 1:]

        # decay: Make sure decay is between 0 and 1
        cluster_decay = scale_up(np.array(serialized_parameters[:self.k + 1]), self.centers[11], self.scales[11])
        serialized_parameters = serialized_parameters[self.k + 1:]
        cluster_decay = np.minimum(1, np.maximum(0, cluster_decay))

        net_params = {
            'N': self.N,
            'mix': mix,
            'mu': mu,
            'variance': var,
            'correlation': correlation,
            'inhibitory': self.inhibitory,
            'connectivity': connectivity,
            'cluster_connectivity': self.cluster_connectivity,
            'weight_scaling': weight_scaling,
            'weight_mean': weight_mean,
            'bias_scaling': bias_scaling,
            'bias_mean': bias_mean,
            'x_lim': self.x_lim,
            'y_lim': self.y_lim,
            'decay': cluster_decay,
            'size_in': self.size_in,
            'size_out': self.size_out,
            'in_loc': self.in_loc,
            'act_func': self.activation_func,
            'dt': self.dt
        }
        return net_params

    def get_new_network_from_serialized(self, serialized_parameters):
        net_params = self.get_parameters_from_serialized(serialized_parameters)
        return GMMPopulation(**net_params)

    def get_serialized_parameters(self):
        # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, weight_mean, bias_scaling, bias_mean,
        # cluster_decay

        def scale_down(params, middle, scale):
            scaled = params - middle
            scaled = scaled / scale
            return scaled

        parameters = [self.mix, self.mu_x, self.mu_y, self.variance_x, self.variance_y, self.correlation,
                      self.connectivity, self.weight_scaling, self.weight_mean, self.bias_scaling, self.bias_mean,
                      self.cluster_decay]
        serialized_parameters = []
        for i, par in enumerate(parameters):
            serialized_parameters += list(scale_down(par.flatten(), self.centers[i], self.scales[i]))

        return np.array(serialized_parameters)


class GMMPopulationAdaptive(DistDelayNetwork):
    """
    Class for a rate-based continuous neuron population that implements time delays based on signal propagation
    distance. Neurons are positioned in clusters, with coordinates sampled from a gaussian mixture. Synaptic plasticity
    based on BCM rule is included.
    """

    def __init__(self, N, mix, mu, variance, correlation, inhibitory, connectivity, cluster_connectivity,
                 weight_scaling, weight_mean, bias_scaling, bias_mean, decay, lr_mean, lr_scaling,
                 y0_mean, y0_scaling, size_in, size_out, in_loc, act_func=sigmoid_activation, autapse=False, dt=0.0001,
                 x_lim=None, y_lim=None, theta_window=None):

        assert (x_lim is None) == (y_lim is None), "For now it is not possible to have only one of the two axes limited"

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.size_in = size_in
        self.size_out = size_out
        self.k = len(mix)
        self.mix = mix
        self.mu_x = mu[:, 0]
        self.mu_y = mu[:, 1]
        self.variance_x = variance[:, 0]
        self.variance_y = variance[:, 1]
        self.correlation = correlation
        self.inhibitory = inhibitory
        self.connectivity = connectivity
        self.cluster_connectivity = cluster_connectivity
        self.weight_scaling = weight_scaling
        self.weight_mean = weight_mean
        self.bias_scaling = bias_scaling
        self.bias_mean = bias_mean
        self.lr_mean = lr_mean
        self.lr_scaling = lr_scaling
        self.y0_mean = y0_mean
        self.y0_scaling = y0_scaling
        self.in_loc = in_loc
        self.autapse=autapse
        self.cluster_decay = decay
        self.fixed = fixed_hyperparameters

        self.ranges = hyperparameter_ranges
        self.centers = {key: (self.ranges[key][1] + self.ranges[key][0]) / 2 for key in self.ranges}
        self.scales = {key: self.ranges[key][1] - self.ranges[key][0] for key in self.ranges}

        self.serialization_order = {'mix': self.mix,
                                    'mu_x': self.mu_x,
                                    'mu_y': self.mu_y,
                                    'variance_x': self.variance_x,
                                    'variance_y': self.variance_y,
                                    'correlation': self.correlation,
                                    'connectivity': self.connectivity,
                                    'weight_scaling': self.weight_scaling,
                                    'weight_mean': self.weight_mean,
                                    'bias_scaling': self.bias_scaling,
                                    'bias_mean': self.bias_mean,
                                    'lr_scaling': self.lr_scaling,
                                    'lr_mean': self.lr_mean,
                                    'y0_scaling': self.y0_scaling,
                                    'y0_mean': self.y0_mean,
                                    'cluster_decay': self.cluster_decay}
        k = self.k
        var_mat = np.zeros((k, 2, 2))
        corr_mat = np.ones((k, 2, 2))
        for i in range(k):
            var_mat[i, 0, 0] = self.variance_x[i]
            var_mat[i, 1, 1] = self.variance_y[i]
            corr_mat[i, 0, 1] = self.correlation[i]
            corr_mat[i, 1, 0] = self.correlation[i]

        Sigma = np.matmul(np.matmul(var_mat, corr_mat), var_mat)
        self.covariances = Sigma
        grid, n_type, clusters = get_gaussian_mixture_config(N - size_in, mix, mu, inhibitory, Sigma,
                                                             x_lim, y_lim)


        mu_x_in, mu_y_in = in_loc

        Sigma_in = 0.002 * (np.array(
            [
                [0, 0],
                [0, 0]
            ])) ** 2
        Sigma_in = np.expand_dims(Sigma_in, 0)

        grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, size_in, mu_x_in, mu_y_in,
                                                   Sigma=Sigma_in,
                                                   x_range=x_lim, y_range=y_lim)
        self.clusters = clusters
        input_index = np.array(range(0, size_in))
        output_index = np.random.choice(range(size_in, N), size=(size_out, 1), replace=False)

        bias = np.zeros((N,))
        n_decay = np.zeros((N,))
        y0 = np.zeros((N,))
        for i in range(N):
            c = int(clusters[i])
            sf = bias_scaling[c]
            bias[i] = np.random.uniform(-0.5, 0.5) * sf + bias_mean[c]
            n_decay[i] = decay[c]
            y0[i] = np.random.normal(y0_mean[c], y0_scaling[c])
        y0 = np.minimum(np.maximum(y0, np.ones_like(y0) * .25), np.ones_like(y0))
        W = get_weight_matrix(weight_mean, weight_scaling, clusters)
        lr = get_lr(clusters, lr_mean, lr_scaling)

        if not autapse:
            np.fill_diagonal(W, 0)  # No self connections
        W = clustered_pop_pruning(W, connectivity, clusters, cluster_connectivity)

        super().__init__(weights=W, bias=bias, n_type=n_type, coordinates=grid, decay=n_decay, input_n=input_index,
                         output_n=output_index, activation_func=act_func, dt=dt,
                         theta_window=theta_window, lr=lr, theta_y0=y0)

    def get_parameters_from_serialized(self, serialized_parameters):
        serialized_parameters = np.array(serialized_parameters)

        def scale_up(params, middle, scale):
            scaled = params * scale
            scaled = scaled + middle
            return scaled

        # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, weight_mean, bias_scaling, bias_mean,
        # lr_scaling, lr_mean, y0_scaling, y0_mean, decay

        # mix: Apply limit at 0 and normalize to get proper distribution
        if not self.fixed['mix']:
            mix = scale_up(np.array(serialized_parameters[:self.k]), self.centers['mix'], self.scales['mix'])
            serialized_parameters = serialized_parameters[self.k:]
            mix = np.maximum(0, mix)
            sum_mix = np.sum(mix)
            if sum_mix != 0:
                mix = mix / np.sum(mix)
            else:
                mix = np.ones_like(mix) / self.k
        else:
            mix = self.mix

        # mu_x: Apply hard limit to not exceed ranges
        if not self.fixed['mu_x']:
            mu_x = scale_up(np.array(serialized_parameters[:self.k]), self.centers['mu_x'], self.scales['mu_x'])
            serialized_parameters = serialized_parameters[self.k:]
            mu_x = np.maximum(mu_x, self.x_range[0])
            mu_x = np.minimum(mu_x, self.x_range[1])
        else:
            mu_x = self.mu_x

        # mu_y: Apply hard limit to not exceed ranges
        if not self.fixed['mu_y']:
            mu_y = scale_up(np.array(serialized_parameters[:self.k]), self.centers['mu_y'], self.scales['mu_y'])
            serialized_parameters = serialized_parameters[self.k:]
            mu_y = np.maximum(mu_y, self.y_range[0])
            mu_y = np.minimum(mu_y, self.y_range[1])
        else:
            mu_y = self.mu_y

        mu = np.zeros((self.k, 2))
        mu[:, 0] = mu_x
        mu[:, 1] = mu_y

        # var_x: Make sure var is non-negative)
        if not self.fixed['variance_x']:
            var_x = scale_up(np.array(serialized_parameters[:self.k]), self.centers['variance_x'],
                             self.scales['variance_x'])
            serialized_parameters = serialized_parameters[self.k:]
            var_x = np.maximum(0, var_x)
        else:
            var_x = self.variance_x

        # var_y: Make sure var is non-negative)
        if not self.fixed['variance_y']:
            var_y = scale_up(np.array(serialized_parameters[:self.k]), self.centers['variance_y'],
                             self.scales['variance_y'])
            serialized_parameters = serialized_parameters[self.k:]
            var_y = np.maximum(0, var_y)
        else:
            var_y = self.variance_y

        var = np.zeros((self.k, 2))
        var[:, 0] = var_x
        var[:, 1] = var_y

        # corr_xy: Make sure correlation is between -1 and 1
        if not self.fixed['correlation']:
            correlation = scale_up(np.array(serialized_parameters[:self.k]), self.centers['correlation'],
                                   self.scales['correlation'])
            serialized_parameters = serialized_parameters[self.k:]
            correlation = np.minimum(np.maximum(correlation, -0.99), 0.99)
        else:
            correlation = self.correlation

        # conn: Make sure connectivity is between 0 and 1
        if not self.fixed['connectivity']:
            connectivity = scale_up(np.array(serialized_parameters[:(self.k + 1) ** 2]),
                                    self.centers['connectivity'],
                                    self.scales['connectivity'])
            connectivity = np.reshape(connectivity, (self.k + 1, self.k + 1))
            serialized_parameters = serialized_parameters[(self.k + 1) ** 2:]
            connectivity = np.minimum(np.maximum(connectivity, 0), 1)
        else:
            connectivity = self.connectivity

        # weight_scaling: Weight scaling should be non-negative
        if not self.fixed['weight_scaling']:
            weight_scaling = scale_up(np.array(serialized_parameters[:(self.k + self.size_in) ** 2]),
                                      self.centers['weight_scaling'],
                                      self.scales['weight_scaling'])
            weight_scaling = np.reshape(weight_scaling, (self.k + self.size_in, self.k + self.size_in))
            serialized_parameters = serialized_parameters[(self.k + self.size_in) ** 2:]
            weight_scaling = np.maximum(weight_scaling, 0)
        else:
            weight_scaling = self.weight_scaling

        # weight_mean
        if not self.fixed['weight_mean']:
            weight_mean = scale_up(np.array(serialized_parameters[:(self.k + self.size_in) ** 2]),
                                   self.centers['weight_mean'],
                                   self.scales['weight_mean'])
            weight_mean = np.reshape(weight_mean, (self.k + self.size_in, self.k + self.size_in))
            serialized_parameters = serialized_parameters[(self.k + self.size_in) ** 2:]
        else:
            weight_mean = self.weight_mean

        # bias_scaling: Bias scaling should be non-negative
        if not self.fixed['bias_scaling']:
            bias_scaling = scale_up(np.array(serialized_parameters[:self.k + self.size_in]),
                                    self.centers['bias_scaling'],
                                    self.scales['bias_scaling'])
            serialized_parameters = serialized_parameters[self.k + self.size_in:]
            bias_scaling = np.maximum(bias_scaling, 0)
        else:
            bias_scaling = self.bias_scaling

        # bias_mean
        if not self.fixed['bias_mean']:
            bias_mean = scale_up(np.array(serialized_parameters[:self.k + self.size_in]),
                                 self.centers['bias_mean'],
                                 self.scales['bias_mean'])
            serialized_parameters = serialized_parameters[self.k + self.size_in:]
        else:
            bias_mean = self.bias_mean

        # lr_scaling: learning rate std should be non-negative
        if not self.fixed['lr_scaling']:
            lr_scaling = scale_up(np.array(serialized_parameters[:(self.k + self.size_in) ** 2]),
                                  self.centers['lr_scaling'],
                                  self.scales['lr_scaling'])
            serialized_parameters = serialized_parameters[(self.k + self.size_in) ** 2:]
            lr_scaling = np.reshape(lr_scaling, (self.k + self.size_in, self.k + self.size_in))
            lr_scaling = np.maximum(lr_scaling, 0)
        else:
            lr_scaling = self.lr_scaling

        # lr_mean: learning rate mean should be non-negative
        if not self.fixed['lr_mean']:
            lr_mean = scale_up(np.array(serialized_parameters[:(self.k + self.size_in) ** 2]),
                               self.centers['lr_mean'],
                               self.scales['lr_mean'])
            serialized_parameters = serialized_parameters[(self.k + self.size_in) ** 2:]
            lr_mean = np.reshape(lr_mean, (self.k + self.size_in, self.k + self.size_in))
            lr_mean = np.maximum(lr_mean, 0)
        else:
            lr_mean = self.lr_mean

        # y0_scaling: y0 std should be non-negative
        if not self.fixed['y0_scaling']:
            y0_scaling = scale_up(np.array(serialized_parameters[:self.k + self.size_in]),
                                  self.centers['y0_scaling'],
                                  self.scales['y0_scaling'])
            serialized_parameters = serialized_parameters[self.k + self.size_in:]
            y0_scaling = np.maximum(y0_scaling, 0)
        else:
            y0_scaling = self.y0_scaling

        # y0_mean: y0 mean should be greater than .25
        if not self.fixed['y0_mean']:
            y0_mean = scale_up(np.array(serialized_parameters[:self.k + self.size_in]),
                               self.centers['y0_mean'],
                               self.scales['y0_mean'])
            serialized_parameters = serialized_parameters[self.k + self.size_in:]
            y0_mean = np.maximum(y0_mean, 0.25)
        else:
            y0_mean = self.y0_mean

        # decay: Make sure decay is between 0 and 1
        if not self.fixed['cluster_decay']:
            cluster_decay = scale_up(np.array(serialized_parameters[:self.k + self.size_in]),
                                     self.centers['cluster_decay'],
                                     self.scales['cluster_decay'])
            serialized_parameters = serialized_parameters[self.k + self.size_in:]
            cluster_decay = np.minimum(1, np.maximum(0, cluster_decay))
        else:
            cluster_decay = self.cluster_decay

        net_params = {
            'N': self.N,
            'mix': mix,
            'mu': mu,
            'variance': var,
            'correlation': correlation,
            'inhibitory': self.inhibitory,
            'connectivity': connectivity,
            'cluster_connectivity': self.cluster_connectivity,
            'weight_scaling': weight_scaling,
            'weight_mean': weight_mean,
            'bias_scaling': bias_scaling,
            'bias_mean': bias_mean,
            'x_lim': self.x_lim,
            'y_lim': self.y_lim,
            'decay': cluster_decay,
            'lr_mean': lr_mean,
            'lr_scaling': lr_scaling,
            'y0_mean': y0_mean,
            'y0_scaling': y0_scaling,
            'size_in': self.size_in,
            'size_out': self.size_out,
            'in_loc': self.in_loc,
            'act_func': self.activation_func,
            'dt': self.dt
        }
        return net_params

    def get_new_network_from_serialized(self, serialized_parameters):
        net_params = self.get_parameters_from_serialized(serialized_parameters)
        return GMMPopulationAdaptive(**net_params)


    def get_serialized_parameters(self):
        # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, weight_mean, bias_scaling, bias_mean,
        # lr_scaling, lr_mean, y0_scaling, y0_mean, decay

        def scale_down(params, middle, scale):
            scaled = params - middle
            scaled = scaled / scale
            return scaled

        serialized_parameters = []

        for key in self.serialization_order:
            if not self.fixed[key]:
                serialized_parameters += list(scale_down(self.serialization_order[key].flatten(),
                                                         self.centers[key],
                                                         self.scales[key]))
        return np.array(serialized_parameters)


class GMMPopulationOld(DistDelayNetwork):
    """
    Class for a rate-based continuous neuron population that implements time delays based on signal propagation
    distance. Neurons are positioned in clusters, with coordinates sampled from a gaussian mixture.
    """

    def __init__(self, N, mix, mu, variance, correlation, inhibitory, connectivity, cluster_connectivity,
                 weight_scaling, bias_scaling, x_range, y_range, decay, size_in, size_out, in_loc,
                 act_func=tanh_activation, autapse=False, dt=0.0001, fixed_delay=0):

        self.size_in = size_in
        self.size_out = size_out
        self.k = len(mix)
        self.mix = mix
        self.mu_x = mu[:, 0]
        self.mu_y = mu[:, 1]
        self.variance_x = variance[:, 0]
        self.variance_y = variance[:, 1]
        self.correlation = correlation
        self.inhibitory = inhibitory
        self.connectivity = connectivity
        self.cluster_connectivity = cluster_connectivity
        self.weight_scaling = weight_scaling
        self.bias_scaling = bias_scaling
        self.in_loc = in_loc
        self.autapse=autapse
        x_range = x_range
        y_range = y_range
        self.cluster_decay = decay
        self.fixed_delay=fixed_delay

        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        r_mix = [0, 1]
        r_corr = [-1, 1]
        r_mu_x = [x_range[0], x_range[1]]
        r_mu_y = [y_range[0], y_range[1]]
        r_variance_x = r_variance_y = [0, (width + height) / 4]
        # r_variance_y = [0, y_range[1] - y_range[0]]
        r_inh = [0, 1]
        r_conn = [0, 1]
        r_decay = [0, 1]
        r_bias_scaling = [0, 2]
        r_weight_scaling = [0, 2]

        self.ranges = [r_mix, r_mu_x, r_mu_y, r_variance_x, r_variance_y, r_corr, r_conn, r_weight_scaling,
                       r_bias_scaling, r_decay]
        self.centers = [(r[1] + r[0]) / 2 for r in self.ranges]
        self.scales = [r[1] - r[0] for r in self.ranges]

        k = self.k
        var_mat = np.zeros((k, 2, 2))
        corr_mat = np.ones((k, 2, 2))
        for i in range(k):
            var_mat[i, 0, 0] = self.variance_x[i]
            var_mat[i, 1, 1] = self.variance_y[i]
            corr_mat[i, 0, 1] = self.correlation[i]
            corr_mat[i, 1, 0] = self.correlation[i]

        Sigma = np.matmul(np.matmul(var_mat, corr_mat), var_mat)

        grid, n_type, clusters = get_gaussian_mixture_config(N - size_in, mix, mu, inhibitory, Sigma,
                                                             x_range, y_range)

        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        mu_x_in, mu_y_in = in_loc

        Sigma_in = 0.002 * (np.array(
            [
                [width, 0],
                [0, height]
            ])) ** 2
        Sigma_in = np.expand_dims(Sigma_in, 0)

        grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, size_in, mu_x_in, mu_y_in,
                                                   Sigma=Sigma_in,
                                                   x_range=x_range, y_range=y_range)

        input_index = np.array(range(0, size_in))
        output_index = np.random.choice(range(size_in, N), size=(size_out, 1), replace=False)

        bias = np.zeros((N,))
        n_decay = np.zeros((N,))
        for i in range(N):
            c = int(clusters[i])
            sf = bias_scaling[c]
            bias[i] = np.random.uniform(-0.5, 0.5) * sf
            n_decay[i] = decay[c]

        W = get_weight_matrix_old(weight_scaling, clusters)
        if not autapse:
            np.fill_diagonal(W, 0)  # No self connections
        W = clustered_pop_pruning(W, connectivity, clusters, cluster_connectivity)

        super().__init__(weights=W, bias=bias, n_type=n_type, coordinates=grid, decay=n_decay, input_n=input_index,
                         output_n=output_index, activation_func=act_func, dt=dt)
        if fixed_delay > 0:
            self.D = np.ones_like(self.D) * fixed_delay

    def get_parameters_from_serialized(self, serialized_parameters):
        serialized_parameters = np.array(serialized_parameters)

        def scale_up(params, middle, scale):
            scaled = params * scale
            scaled = scaled + middle
            return scaled

        # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, bias_scaling, decay

        # mix: Apply limit at 0 and normalize to get proper distribution
        mix = scale_up(np.array(serialized_parameters[:self.k]), self.centers[0], self.scales[0])
        serialized_parameters = serialized_parameters[self.k:]
        mix = np.maximum(0, mix)
        sum_mix = np.sum(mix)
        if sum_mix != 0:
            mix = mix / np.sum(mix)
        else:
            mix = np.ones_like(mix)

        # mu_x: Apply hard limit to not exceed ranges
        mu_x = scale_up(np.array(serialized_parameters[:self.k]), self.centers[1], self.scales[1])
        serialized_parameters = serialized_parameters[self.k:]
        mu_x = np.maximum(mu_x, self.x_range[0])
        mu_x = np.minimum(mu_x, self.x_range[1])

        # mu_y: Apply hard limit to not exceed ranges
        mu_y = scale_up(np.array(serialized_parameters[:self.k]), self.centers[2], self.scales[2])
        serialized_parameters = serialized_parameters[self.k:]
        mu_y = np.maximum(mu_y, self.y_range[0])
        mu_y = np.minimum(mu_y, self.y_range[1])

        mu = np.zeros((self.k, 2))
        mu[:, 0] = mu_x
        mu[:, 1] = mu_y

        # var_x: Make sure var is non-negative)
        var_x = scale_up(np.array(serialized_parameters[:self.k]), self.centers[3], self.scales[3])
        serialized_parameters = serialized_parameters[self.k:]
        var_x = np.maximum(0, var_x)

        # var_y: Make sure var is non-negative)
        var_y = scale_up(np.array(serialized_parameters[:self.k]), self.centers[4], self.scales[4])
        serialized_parameters = serialized_parameters[self.k:]
        var_y = np.maximum(0, var_y)

        var = np.zeros((self.k, 2))
        var[:, 0] = var_x
        var[:, 1] = var_y

        # corr_xy: Make sure correlation is between -1 and 1
        correlation = scale_up(np.array(serialized_parameters[:self.k]), self.centers[5], self.scales[5])
        serialized_parameters = serialized_parameters[self.k:]
        correlation = np.minimum(np.maximum(correlation, -0.99), 0.99)

        # conn: Make sure connectivity is between 0 and 1
        connectivity = scale_up(np.array(serialized_parameters[:(self.k + 1) ** 2]), self.centers[6], self.scales[6])
        connectivity = np.reshape(connectivity, (self.k + 1, self.k + 1))
        serialized_parameters = serialized_parameters[(self.k + 1) ** 2:]
        connectivity = np.minimum(np.maximum(connectivity, 0), 1)

        # weight_scaling: Weight scaling should be non-negative
        weight_scaling = scale_up(np.array(serialized_parameters[:(self.k + 1) ** 2]), self.centers[7], self.scales[7])
        weight_scaling = np.reshape(weight_scaling, (self.k + 1, self.k + 1))
        serialized_parameters = serialized_parameters[(self.k + 1) ** 2:]
        weight_scaling = np.maximum(weight_scaling, 0)

        # bias_scaling: Bias scaling should be non-negative
        bias_scaling = scale_up(np.array(serialized_parameters[:self.k + 1]), self.centers[8], self.scales[8])
        serialized_parameters = serialized_parameters[self.k + 1:]
        bias_scaling = np.maximum(bias_scaling, 0)

        # decay: Make sure decay is between 0 and 1
        cluster_decay = scale_up(np.array(serialized_parameters[:self.k + 1]), self.centers[9], self.scales[9])
        serialized_parameters = serialized_parameters[self.k + 1:]
        cluster_decay = np.minimum(1, np.maximum(0, cluster_decay))

        net_params = {
            'N': self.N,
            'mix': mix,
            'mu': mu,
            'variance': var,
            'correlation': correlation,
            'inhibitory': self.inhibitory,
            'connectivity': connectivity,
            'cluster_connectivity': self.cluster_connectivity,
            'weight_scaling': weight_scaling,
            'bias_scaling': bias_scaling,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'decay': cluster_decay,
            'size_in': self.size_in,
            'size_out': self.size_out,
            'in_loc': self.in_loc,
            'act_func': self.activation_func,
            'dt': self.dt
        }
        return net_params

    def get_new_network_from_serialized(self, serialized_parameters):
        net_params = self.get_parameters_from_serialized(serialized_parameters)
        return GMMPopulationOld(**net_params)

    def get_serialized_parameters(self):
        # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, bias_scaling, cluster_decay

        def scale_down(params, middle, scale):
            scaled = params - middle
            scaled = scaled / scale
            return scaled

        parameters = [self.mix, self.mu_x, self.mu_y, self.variance_x, self.variance_y, self.correlation,
                      self.connectivity, self.weight_scaling, self.bias_scaling, self.cluster_decay]
        serialized_parameters = []
        for i, par in enumerate(parameters):
            serialized_parameters += list(scale_down(par.flatten(), self.centers[i], self.scales[i]))

        return np.array(serialized_parameters)


def scale_up(params, middle, scale):
    scaled = params * scale
    scaled = scaled + middle
    return scaled


# static functions
def get_empty_parameters(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    start_mix = np.ones((k,))/k
    start_mu = np.zeros((k, 2))
    start_var = np.zeros((k, 2))
    start_corr = np.zeros((k,))
    inhib_start = np.zeros((k,))
    conn_start = np.zeros((k+1, k+1))
    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.zeros((k + 1,))
    bias_mean = np.zeros((k + 1,))
    weight_scaling = np.ones((k + 1, k + 1))
    weight_mean = np.zeros((k + 1, k + 1))
    decay_start = np.ones((k + 1,))
    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': start_var,
        'correlation': start_corr,
        'inhibitory': inhib_start,
        'connectivity': conn_start,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling,
        'weight_mean': weight_mean,
        'bias_scaling': bias_scaling,
        'bias_mean': bias_mean,
        'x_range': x_range,
        'y_range': y_range,
        'decay': decay_start,
        'size_in': insize,
        'size_out': N-insize,
        'in_loc': in_loc,
        'act_func': activation_function,
        'dt': dt,
        'buffersize': buffersize
    }
    return net_params

def get_empty_parameters_old(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    start_mix = np.ones((k,))/k
    start_mu = np.zeros((k, 2))
    start_var = np.zeros((k, 2))
    start_corr = np.zeros((k,))
    inhib_start = np.zeros((k,))
    conn_start = np.zeros((k+1, k+1))
    cluster_connectivity = np.ones((k + 1, k + 1))
    bias_scaling = np.zeros((k + 1,))
    # bias_mean = np.zeros((k + 1,))
    weight_scaling = np.ones((k + 1, k + 1))
    # weight_mean = np.zeros((k + 1, k + 1))
    decay_start = np.ones((k + 1,))
    net_params = {
        'N': N,
        'mix': start_mix,
        'mu': start_mu,
        'variance': start_var,
        'correlation': start_corr,
        'inhibitory': inhib_start,
        'connectivity': conn_start,
        'cluster_connectivity': cluster_connectivity,
        'weight_scaling': weight_scaling,
        # 'weight_mean': weight_mean,
        'bias_scaling': bias_scaling,
        # 'bias_mean': bias_mean,
        'x_range': x_range,
        'y_range': y_range,
        'decay': decay_start,
        'size_in': insize,
        'size_out': N-insize,
        'in_loc': in_loc,
        'act_func': activation_function,
        'dt': dt,
        'buffersize': buffersize
    }
    return net_params



# def get_empty_GMMpop(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
#     params = get_empty_parameters(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc)
#     return GMMPopulation(**params)

def get_empty_GMMpop_Old(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    params = get_empty_parameters_old(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc)
    return GMMPopulationOld(**params)


def get_weight_matrix_old(scaling, clusters):
    N = clusters.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cl_pre = int(clusters[i])
            cl_post = int(clusters[j])
            sf = scaling[cl_pre, cl_post]
            W[i, j] = np.random.uniform(-.5, .5) * sf
    return W

def get_empty_adaptive_parameters(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    params = get_empty_parameters(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc)
    lr_mean = np.ones((k+1, k+1)) * .1
    lr_scaling = np.ones((k+1, k+1)) * .1
    y0_mean = np.ones((k+1,))
    y0_scaling = np.ones((k+1,)) * .1
    params['lr_mean'] = lr_mean
    params['lr_scaling'] = lr_scaling
    params['y0_mean'] = y0_mean
    params['y0_scaling'] = y0_scaling
    return params


def get_empty_adaptive_GMMpop(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    params = get_empty_adaptive_parameters(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc)
    return GMMPopulationAdaptive(**params)

def get_weight_matrix(mean, scaling, clusters):
    N = clusters.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cl_pre = int(clusters[i])
            cl_post = int(clusters[j])
            sf = scaling[cl_pre, cl_post]
            m = mean[cl_pre, cl_post]
            W[i, j] = np.random.uniform(-.5, .5) * sf + m
    return W


def get_weight_matrix_constrained(scaling, clusters):
    N = clusters.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cl_pre = int(clusters[i])
            cl_post = int(clusters[j])
            sf = scaling[cl_pre, cl_post]
            W[i, j] = np.random.uniform(0, 1) * sf
    return W


def get_lr(clusters, lr_mean, lr_scaling):
    N = len(clusters)
    lr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cl_pre = int(clusters[i])
            cl_post = int(clusters[j])
            lr[i, j] = np.random.normal(lr_mean[cl_pre, cl_post], lr_scaling[cl_pre, cl_post])
    lr = np.maximum(lr, np.zeros_like(lr))
    return lr.T


def get_gaussian_mixture_config(N, w, mu, inhibitory, Sigma=None, x_range=None, y_range=None):
    if Sigma is None:
        Sigma = np.array([
            [0.3, 0],
            [0, 0.3]
        ])
        Sigma = np.expand_dims(Sigma, 0)

    nr_of_gaussians = w.shape[0]
    k = np.random.choice(list(range(nr_of_gaussians)), size=(N,), p=w)
    n_type = np.ones((N,))
    grid = np.zeros((N, 2))
    clusters = np.zeros(N, )

    for i in range(N):
        gaussian = k[i]
        clusters[i] = gaussian
        p_inhib = inhibitory[gaussian]
        if p_inhib > np.random.uniform():
            n_type[i] = -1
        mean = mu[gaussian, :]
        cov = Sigma[gaussian, :, :]
        # cov = np.reshape(cov, (cov.shape[1], cov.shape[2]))
        new_N = np.random.multivariate_normal(mean, cov)
        if x_range is not None and y_range is not None:
            lower = np.array([x_range[0], y_range[0]])
            upper = np.array([x_range[1], y_range[1]])
            grid[i, :] = np.minimum(np.maximum(new_N, lower), upper)
        else:
            grid[i, :] = new_N

    return grid, n_type, clusters


def set_inout_cluster(grid, ntype, clusters, N, x=1, y=1, inhib=0.0, Sigma=None, x_range=None, y_range=None):
    new_grid = np.zeros((N + grid.shape[0], 2))
    new_ntype = np.ones((N + ntype.shape[0],))
    new_clusters = np.zeros((N + ntype.shape[0],))
    mu = np.zeros((1, 2))
    mu[0, 0] = x
    mu[0, 1] = y

    inout_grid, inout_ntype, inout_clusters = get_gaussian_mixture_config(N, np.array([1]), mu, np.array([inhib]),
                                                                          Sigma=Sigma, x_range=x_range, y_range=y_range)
    # for i in range(N):
    #     new_grid[i, :] = np.array([x + i * 0.4, y])
    #     max_index = i
    new_grid[:N, :] = inout_grid
    new_grid[N:, :] = grid
    new_ntype[:N] = inout_ntype
    new_ntype[N:] = ntype
    cluster_inds = list(range(int(np.max(clusters)), int(np.max(clusters)) + N))
    new_clusters[:N] = cluster_inds #np.max(clusters) + 1
    new_clusters[N:] = clusters
    return new_grid, new_ntype, new_clusters


def clustered_pop_pruning(W, connectivity, clusters, cluster_connectivity):
    assert clusters.shape[0] == W.shape[0] == W.shape[1], 'Weight matrix should be N by N and cluster assignment ' \
                                                          'should be N by 1 '
    assert np.max(clusters) + 1 <= connectivity.shape[0], 'number of clusters should match connectivity matrix size. '
    assert connectivity.shape[0] == connectivity.shape[1], 'Connectivity matrix should be square'
    assert cluster_connectivity.shape[0] == cluster_connectivity.shape[1]

    N = W.shape[0]
    for n1 in range(N):
        for n2 in range(N):
            cn1 = int(clusters[n1])
            cn2 = int(clusters[n2])
            cl_conn = cluster_connectivity[cn1, cn2]
            p_prune = (1 - connectivity[cn1, cn2])
            if (np.random.uniform() <= p_prune) or (cl_conn < 1):
                W[n1, n2] = 0
    return W

