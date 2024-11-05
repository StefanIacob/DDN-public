import random

import numpy as np
from numpy.f2py.auxfuncs import throw_error

from network import DistDelayNetwork, DistDelayNetworkOld, DistDelayNetworkSpiking, tanh_activation, sigmoid_activation
import copy


# from pureples.shared.visualize import draw_es


class Evolvable(object):
    def __init__(self, param_dict, ranges_dict, limits_dict):
        assert set(param_dict.keys()) == set(ranges_dict.keys()) == set(limits_dict.keys())
        self.params = param_dict
        self.shapes = {k: np.array(self.params[k]).shape for k in self.params}
        self.ranges = ranges_dict
        self.limits = limits_dict

        self.centers = {k: (self.ranges[k][1] + self.ranges[k][0]) / 2 for k in self.ranges}
        self.scales = {k: self.ranges[k][1] - self.ranges[k][0] for k in self.ranges}

        self.serialized_size = int(np.sum([np.prod(self.shapes[k]) for k in self.shapes]))

    def get_parameters_from_serialized(self, serialized_parameters):
        assert len(serialized_parameters) == self.serialized_size

        def scale_up(params, middle, scale):
            scaled = params * scale
            scaled = scaled + middle
            return scaled

        parameter_stack = np.array(serialized_parameters)
        new_parameters = {}
        for param_name in self.shapes:
            target_shape = self.shapes[param_name]
            flat_shape = int(np.prod(target_shape))
            param_flat = scale_up(parameter_stack[:flat_shape], self.centers[param_name], self.scales[param_name])
            param_flat = np.clip(param_flat, self.limits[param_name][0], self.limits[param_name][1])
            parameter_stack = parameter_stack[flat_shape:]
            param = np.reshape(param_flat, target_shape)
            new_parameters[param_name] = param

        return new_parameters

    def get_serialized_parameters(self):
        def scale_down(params, middle, scale):
            scaled = params - middle
            scaled = scaled / scale
            return scaled

        serialized_parameters = []
        for key in self.params:
            serialized_param = list(
                scale_down(np.array(self.params[key]).flatten(), self.centers[key], self.scales[key]))
            serialized_parameters += serialized_param
        # sanity check
        assert len(serialized_parameters) == self.serialized_size
        return np.array(serialized_parameters)


class NeuronType(Evolvable):

    def __init__(self, p_inhibitory, bias_mean, bias_scaling, decay):
        self.p_inhibitory = p_inhibitory
        self.decay = decay
        self.bias_mean = bias_mean
        self.bias_scaling = bias_scaling
        param_dict = copy.deepcopy(self.__dict__)
        r_inh = [0, 1]
        r_decay = [0, 1]
        r_bias_scaling = [0, 2]
        r_bias_mean = [-1, 1]

        self.ranges = {
            'p_inhibitory': r_inh,
            'decay': r_decay,
            'bias_mean': r_bias_mean,
            'bias_scaling': r_bias_scaling
        }

        self.limits = {
            'p_inhibitory': r_inh,
            'decay': r_decay,
            'bias_mean': [-float('inf'), float('inf')],
            'bias_scaling': [0, float('inf')]
        }

        super().__init__(param_dict, self.ranges, self.limits)

    def get_neurons(self, N):
        ex_in = np.random.choice([-1, 1], size=N, replace=True, p=[self.p_inhibitory, 1 - self.p_inhibitory])
        bias = np.random.normal(loc=self.bias_mean, scale=self.bias_scaling, size=N)
        decay = np.repeat(self.decay, N)
        return ex_in, bias, decay

    def new_n_type_from_serialized(self, serialized_params):
        params = self.get_parameters_from_serialized(serialized_params)
        return NeuronType(**params)


class GeometricNeuronType(NeuronType):

    def __init__(self, p_inhibitory, bias_mean, bias_scaling, decay, projection_length, receptor_strength,
                 terminal_strength):
        self.projection_length = projection_length
        self.receptor_strength = receptor_strength
        self.terminal_strength = terminal_strength

        super().__init__(p_inhibitory, bias_mean, bias_scaling, decay)
        self.ranges['projection_length'] = [0, 1]
        self.ranges['receptor_strength'] = [0, 1]
        self.ranges['terminal_strength'] = [0, 1]

        self.limits['projection_length'] = [0, float('inf')]
        self.limits['receptor_strength'] = [0, 1]
        self.limits['terminal_strength'] = [0, 1]


class GeometricCluster(Evolvable):
    def __init__(self, neuron_type_dist, mu_x, mu_y, variance_x, variance_y, correlation, x_lim=None, y_lim=None):
        assert (x_lim is None) == (y_lim is None), "For now it is not possible to have only one of the two axes limited"
        assert len(np.array(neuron_type_dist).shape) == 1
        assert np.round(np.sum(neuron_type_dist), 8) == 1
        self.neuron_type_dist = neuron_type_dist
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.correlation = correlation

        param_dict = copy.deepcopy(self.__dict__)

        if x_lim is None:
            start_width = 2
            start_height = 2
            r_mu_x = [-1, 1]
            r_mu_y = [-1, 1]
            x_lim = [-float('inf'), float('inf')]
            y_lim = [-float('inf'), float('inf')]
        else:
            start_width = x_lim[1] - x_lim[0]
            start_height = y_lim[1] - y_lim[0]
            r_mu_x = [x_lim[0], x_lim[1]]
            r_mu_y = [y_lim[0], y_lim[1]]

        r_variance_x = [0, start_width / 2]
        r_variance_y = [0, start_height / 2]

        self.ranges = {
            'neuron_type_dist': [0, 1],
            'mu_x': r_mu_x,
            'mu_y': r_mu_y,
            'variance_x': r_variance_x,
            'variance_y': r_variance_y,
            'correlation': [-1, 1],
        }

        self.limits = {
            'neuron_type_dist': [0, float('inf')],
            'mu_x': x_lim,
            'mu_y': y_lim,
            'variance_x': [0, float('inf')],
            'variance_y': [0, float('inf')],
            'correlation': [-1, 1],
        }

        super().__init__(param_dict, self.ranges, self.limits)

    def get_neurons(self, N):
        N_types = np.random.choice(range(len(self.neuron_type_dist)), size=N, replace=True, p=self.neuron_type_dist)
        locations = np.random.multivariate_normal(size=N, mean=self.mu, cov=self.Sigma)
        cluster_weights = self.get_internal_cluster_weights(N, N_types)
        return N_types, locations, cluster_weights

    def new_cluster_from_serialized(self, serialized_params):
        new_params = self.get_parameters_from_serialized(serialized_params)
        new_params['neuron_type_dist'] = new_params['neuron_type_dist'] / np.sum(new_params['neuron_type_dist'])
        return Cluster(**new_params)


class GeometricClusterNet(Evolvable):

    def __init__(self, mixture, nr_n_types, type_connectivity):
        assert np.round(np.sum(mixture), decimals=5) == 1
        assert type_connectivity.shape == nr_n_types
        assert set(np.unique(type_connectivity)) == {0, 1}

        self.mixture = mixture

        param_dict = copy.deepcopy(self.__dict__)

        ranges = {
            'mixture': [0, 1],
            'connectivity': [0, 1],
            'weight_scaling': [0, 2],
            'weight_mean': [0, 2]
        }

        limits = {
            'mixture': [0, 1],
            'connectivity': [0, 1],
            'weight_scaling': [0, float('inf')],
            'weight_mean': [0, float('inf')]
        }

        super().__init__(param_dict, ranges, limits)

    def weights_c2c(self, ci, cj, N, M):
        connectivity = self.connectivity[ci, cj]
        weight_mean = self.weight_mean[ci, cj]
        weight_scaling = self.weight_scaling[ci, cj]
        W = np.zeros(shape=(N, M))
        for i in range(N):
            for j in range(M):
                if i != j:
                    if np.random.uniform(0, 1) < connectivity:
                        W[i, j] = np.random.normal(loc=weight_mean, scale=weight_scaling)

        return np.clip(W, self.limits['weight_mean'][0], self.limits['weight_mean'][1])

    def get_weights(self, cluster_weights_list):
        N_per_cluster = [c.shape[0] for c in cluster_weights_list]
        N_total = sum(N_per_cluster)
        W = np.zeros(shape=(N_total, N_total))
        start_i = 0
        start_j = 0
        for i, N in enumerate(N_per_cluster):
            end_i = start_i + N
            for j, M in enumerate(N_per_cluster):
                end_j = start_j + M
                if i == j:
                    W[start_i:end_i, start_j:end_j] = cluster_weights_list[i]
                else:
                    W_ij = self.weights_c2c(i, j, N, M)
                    W[start_i:end_i, start_j:end_j] = W_ij
                start_j = end_j
            start_j = 0
            start_i = end_i
        return W

    def new_cluster_net_from_serialized(self, serialized_params):
        new_params = self.get_parameters_from_serialized(serialized_params)
        # apply mixture normalization
        new_params['mixture'] = new_params['mixture'] / np.sum(new_params['mixture'])
        return ClusterNet(**new_params)


class Cluster(Evolvable):
    def __init__(self, neuron_type_dist, mu_x, mu_y, variance_x, variance_y, correlation, connectivity, weight_scaling,
                 weight_mean, x_lim=None, y_lim=None):
        assert (x_lim is None) == (y_lim is None), "For now it is not possible to have only one of the two axes limited"
        assert len(np.array(neuron_type_dist).shape) == 1
        assert np.round(np.sum(neuron_type_dist), 8) == 1
        assert connectivity.shape == weight_scaling.shape == weight_mean.shape

        self.neuron_type_dist = neuron_type_dist
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.correlation = correlation
        self.connectivity = connectivity
        self.weight_scaling = weight_scaling
        self.weight_mean = weight_mean
        param_dict = copy.deepcopy(self.__dict__)


        if x_lim is None:
            start_width = 2
            start_height = 2
            r_mu_x = [-1, 1]
            r_mu_y = [-1, 1]
            x_lim = [-float('inf'), float('inf')]
            y_lim = [-float('inf'), float('inf')]
        else:
            start_width = x_lim[1] - x_lim[0]
            start_height = y_lim[1] - y_lim[0]
            r_mu_x = [x_lim[0], x_lim[1]]
            r_mu_y = [y_lim[0], y_lim[1]]

        r_variance_x = [0, start_width / 2]
        r_variance_y = [0, start_height / 2]

        self.ranges = {
            'neuron_type_dist': [0, 1],
            'mu_x': r_mu_x,
            'mu_y': r_mu_y,
            'variance_x': r_variance_x,
            'variance_y': r_variance_y,
            'correlation': [-1, 1],
            'connectivity': [0, 1],
            'weight_scaling': [0, 2],
            'weight_mean': [0, 2],
        }

        self.limits = {
            'neuron_type_dist': [0, float('inf')],
            'mu_x': x_lim,
            'mu_y': y_lim,
            'variance_x': [0, float('inf')],
            'variance_y': [0, float('inf')],
            'correlation': [-1, 1],
            'connectivity': [0, 1],
            'weight_scaling': [0, float('inf')],
            'weight_mean': [0, float('inf')],
        }

        super().__init__(param_dict, self.ranges, self.limits)

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.space_lims = np.array([x_lim, y_lim])

        var_mat = np.array(
            [
                [self.variance_x, 0],
                [0, self.variance_y]
            ]
        )
        corr_mat = np.array(
            [
                [1, self.correlation],
                [self.correlation, 1]
            ]
        )

        self.Sigma = np.matmul(np.matmul(var_mat, corr_mat), var_mat)
        self.mu = np.array([mu_x, mu_y])

    def get_internal_cluster_weights(self, N, N_types=None):
        if N_types is None:
            assert self.connectivity.shape == ()
            W = np.random.uniform(size=(N, N),
                                  low=self.weight_mean - 0.5 * self.weight_scaling,
                                  high=self.weight_mean + 0.5 * self.weight_scaling)
            np.fill_diagonal(W, 0)
            for i in N:
                for j in N:
                    if i != j:
                        p = random.uniform(0, 1)
                        print(p)
            return W

        else:
            assert len(N_types) == N
            W = np.zeros(shape=(N, N))
            for i, n_i in enumerate(N_types):
                for j, n_j in enumerate(N_types):
                    if i != j:
                        p_connection = self.connectivity[n_i, n_j]
                        weight_mean = self.weight_mean[n_i, n_j]
                        weight_scaling = self.weight_scaling[n_i, n_j]
                        if np.random.uniform(0, 1) < p_connection:
                            W[i, j] = np.random.normal(loc=weight_mean, scale=weight_scaling)
            return np.clip(W, self.limits['weight_mean'][0], self.limits['weight_mean'][1])

    def get_neurons(self, N):
        N_types = np.random.choice(range(len(self.neuron_type_dist)), size=N, replace=True, p=self.neuron_type_dist)
        locations = np.random.multivariate_normal(size=N, mean=self.mu, cov=self.Sigma)
        locations = np.clip(locations, self.space_lims[:, 0], self.space_lims[:, 1])
        cluster_weights = self.get_internal_cluster_weights(N, N_types)
        return N_types, locations, cluster_weights

    def new_cluster_from_serialized(self, serialized_params):
        new_params = self.get_parameters_from_serialized(serialized_params)
        if np.sum(new_params['neuron_type_dist']) != 0:
            new_params['neuron_type_dist'] = new_params['neuron_type_dist'] / np.sum(new_params['neuron_type_dist'])
        else:
            new_params['neuron_type_dist'] = np.ones_like(new_params['neuron_type_dist']) / len(new_params['neuron_type_dist'])
        new_params['x_lim'] = self.x_lim
        new_params['y_lim'] = self.y_lim
        return Cluster(**new_params)


class ClusterNet(Evolvable):

    def __init__(self, mixture, connectivity, weight_mean, weight_scaling):
        assert np.round(np.sum(mixture), decimals=5) == 1, 'mixture should sum to 1'

        self.mixture = mixture
        self.connectivity = connectivity
        self.weight_mean = weight_mean
        self.weight_scaling = weight_scaling
        param_dict = copy.deepcopy(self.__dict__)

        ranges = {
            'mixture': [0, 1],
            'connectivity': [0, 1],
            'weight_scaling': [0, 2],
            'weight_mean': [0, 2]
        }

        limits = {
            'mixture': [0, 1],
            'connectivity': [0, 1],
            'weight_scaling': [0, float('inf')],
            'weight_mean': [0, float('inf')]
        }

        super().__init__(param_dict, ranges, limits)

    def weights_c2c(self, ci, cj, N, M):
        connectivity = self.connectivity[ci, cj]
        weight_mean = self.weight_mean[ci, cj]
        weight_scaling = self.weight_scaling[ci, cj]
        W = np.zeros(shape=(N, M))
        for i in range(N):
            for j in range(M):
                if i != j:
                    if np.random.uniform(0, 1) < connectivity:
                        W[i, j] = np.random.normal(loc=weight_mean, scale=weight_scaling)

        return np.clip(W, self.limits['weight_mean'][0], self.limits['weight_mean'][1])

    def get_weights(self, cluster_weights_list):
        N_per_cluster = [c.shape[0] for c in cluster_weights_list]
        N_total = sum(N_per_cluster)
        W = np.zeros(shape=(N_total, N_total))
        start_i = 0
        start_j = 0
        for i, N in enumerate(N_per_cluster):
            end_i = start_i + N
            for j, M in enumerate(N_per_cluster):
                end_j = start_j + M
                if i == j:
                    W[start_i:end_i, start_j:end_j] = cluster_weights_list[i]
                else:
                    W_ij = self.weights_c2c(i, j, N, M)
                    W[start_i:end_i, start_j:end_j] = W_ij
                start_j = end_j
            start_j = 0
            start_i = end_i
        return W

    def new_cluster_net_from_serialized(self, serialized_params):
        new_params = self.get_parameters_from_serialized(serialized_params)
        # apply mixture normalization
        if np.sum(new_params['mixture']) != 0:
            new_params['mixture'] = new_params['mixture'] / np.sum(new_params['mixture'])
        else:
            new_params['mixture'] = np.ones_like(new_params['mixture']) / len(new_params['mixture'])
        return ClusterNet(**new_params)


class EvolvablePopulation(DistDelayNetwork):
    def __init__(self, N, dt, cluster_net, cluster_list, n_type_list, neurons_in=None, neurons_out=None,
                 treat_input_as_cluster=True):
        assert len(cluster_list[0].neuron_type_dist) == len(n_type_list)
        self.treat_input_as_cluster = treat_input_as_cluster
        self.mixture = cluster_net.mixture
        if treat_input_as_cluster:
            assert len(cluster_list) == len(self.mixture) + 1

        self.N = N
        self.cluster_net = cluster_net
        self.cluster_list = cluster_list
        self.n_type_list = n_type_list

        self.neurons_in = neurons_in
        self.neurons_out = neurons_out

        if neurons_in is None:
            self.neurons_in = np.array([0])
        if neurons_out is None:
            self.neurons_out = np.array(range(1, N))

        W, bias, inhib, decay, locs = self.get_network(treat_input_as_cluster)
        super().__init__(W, bias, inhib, locs, decay, activation_func=sigmoid_activation, input_n=self.neurons_in,
                         output_n=self.neurons_out, dt=dt)

    def get_network(self, input_as_cluster):
        # define network parameter variables
        c_weights_list = []
        locs = []
        n_types = []
        inhib = []
        bias = []
        decay = []

        n_clusters = len(self.cluster_list)

        res_neuron_ind = list(set(range(self.N)) - set(self.neurons_in))  # indices of all non-input neurons
        N_res_neurons = len(res_neuron_ind)
        cluster_inds = range(n_clusters)

        # define cluster assignment
        if input_as_cluster:
            cluster_assignment = np.zeros(shape=(self.N,))
            cluster_assignment[res_neuron_ind] = np.random.choice(cluster_inds[:-1], size=N_res_neurons, replace=True,
                                                                  p=self.mixture)
            cluster_assignment[self.neurons_in] = cluster_inds[-1]

        else:
            cluster_assignment = np.random.choice(cluster_inds, size=self.N, replace=True, p=self.mixture)
            cluster_assignment = np.sort(cluster_assignment)

        sort_key = np.argsort(cluster_assignment)
        inv_sort_key = np.argsort(sort_key)
        self.neurons_in = inv_sort_key[self.neurons_in]
        self.neurons_out = inv_sort_key[self.neurons_out]
        cluster_assignment = cluster_assignment[sort_key]

        N_per_cluster = [np.sum(cluster_assignment == i) for i in cluster_inds]

        # define intra-cluster network
        for i, cluster in enumerate(self.cluster_list):
            cluster_n_types, cluster_locations, cluster_weights = cluster.get_neurons(N_per_cluster[i])
            c_weights_list.append(cluster_weights)
            locs.append(cluster_locations)
            n_types.append(cluster_n_types)

        n_types = np.concatenate(n_types)
        inhib_per_type = []
        bias_per_type = []
        decay_per_type = []
        for i in range(len(self.n_type_list)):
            type_amount = np.sum(n_types == i)
            type_i_inhib, type_i_bias, type_i_decay = self.n_type_list[i].get_neurons(type_amount)
            inhib_per_type.append(list(type_i_inhib))
            bias_per_type.append(list(type_i_bias))
            decay_per_type.append(list(type_i_decay))

        for ntype in n_types:
            inhib.append(inhib_per_type[ntype].pop())
            bias.append(bias_per_type[ntype].pop())
            decay.append(decay_per_type[ntype].pop())
        locs = np.concatenate(locs)

        # define inter-cluster network
        W = self.cluster_net.get_weights(c_weights_list)

        return W, np.array(bias), np.array(inhib), np.array(decay), locs

    def get_serialized_network_parameters(self):
        # order of network components: ClusterNet, Clusters, NeuronTypes
        cluster_net_serialized = self.cluster_net.get_serialized_parameters()
        clusters_serialized = [cluster.get_serialized_parameters() for cluster in self.cluster_list]
        clusters_serialized = np.concatenate(clusters_serialized)
        neuron_types_serialized = [neuron_type.get_serialized_parameters() for neuron_type in self.n_type_list]
        neuron_types_serialized = np.concatenate(neuron_types_serialized)
        serialized_network_parameters = np.concatenate([
            cluster_net_serialized,
            clusters_serialized,
            neuron_types_serialized
        ])

        return serialized_network_parameters

    def get_parameters_from_serialized(self, serialized_params):
        # expected order of network components: ClusterNet, Clusters, NeuronTypes
        expected_param_sizes = {
            'ClusterNet': self.cluster_net.serialized_size,
            'Cluster_list': [cluster.serialized_size for cluster in self.cluster_list],
            'NeuronType_list': [neuron_type.serialized_size for neuron_type in self.n_type_list]
        }
        expected_total_size = np.sum([np.sum(expected_param_sizes[key]) for key in expected_param_sizes])
        assert expected_total_size == len(serialized_params)

        cluster_net_params = serialized_params[:expected_param_sizes['ClusterNet']]
        serialized_params = serialized_params[expected_param_sizes['ClusterNet']:]
        new_cluster_net = self.cluster_net.new_cluster_net_from_serialized(cluster_net_params)

        new_cluster_list = []
        for i, p_size in enumerate(expected_param_sizes['Cluster_list']):
            cluster_params = serialized_params[:p_size]
            serialized_params = serialized_params[p_size:]
            new_cluster = self.cluster_list[i].new_cluster_from_serialized(cluster_params)
            new_cluster_list.append(new_cluster)

        new_neuron_type_list = []
        for i, p_size in enumerate(expected_param_sizes['NeuronType_list']):
            n_type_params = serialized_params[:p_size]
            serialized_params = serialized_params[p_size:]
            new_neuron_type = self.n_type_list[i].new_n_type_from_serialized(n_type_params)
            new_neuron_type_list.append(new_neuron_type)

        new_pop_parameters = {
            'N': self.N,
            'dt': self.dt,
            'cluster_net': new_cluster_net,
            'cluster_list': new_cluster_list,
            'n_type_list': new_neuron_type_list,
            'neurons_in': self.neurons_in,
            'neurons_out': self.neurons_out,
            'treat_input_as_cluster': self.treat_input_as_cluster
        }
        return new_pop_parameters

    def get_new_evolvable_population_from_serialized(self, serialized_parameters):
        params = self.get_parameters_from_serialized(serialized_parameters)
        return EvolvablePopulation(**params)


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
        self.autapse = autapse
        self.cluster_decay = decay
        self.fixed_delay = fixed_delay

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

        r_variance_x = [0, start_width / 2]
        r_variance_y = [0, start_height / 2]

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
        self.cluster_assignment = np.array(clusters, dtype='int32')
        super().__init__(weights=W, bias=bias, ex_in=n_type, decay=n_decay, input_n=input_index,
                         output_n=output_index, activation_func=act_func, dt=dt, coordinates=grid)
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


class GMMPopulationSpiking(DistDelayNetworkSpiking):
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
        self.autapse = autapse
        self.cluster_decay = decay
        self.fixed_delay = fixed_delay

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

        r_variance_x = [0, start_width / 2]
        r_variance_y = [0, start_height / 2]

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
        self.cluster_assignment = np.array(clusters, dtype='int32')
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


class ArmPopulation(DistDelayNetwork):
    """
    Class for a rate-based continuous neuron population that implements time delays based on signal propagation
    distance. Neurons are positioned in clusters, with coordinates sampled from a gaussian mixture.
    """

    def __init__(self, N, mix, mu, variance, correlation, inhibitory, connectivity, cluster_connectivity,
                 weight_scaling, weight_mean, bias_scaling, bias_mean, decay, size_in, size_out, in_loc, out_loc,
                 act_func=tanh_activation, autapse=False, dt=0.0001, x_lim=None,
                 y_lim=None, fixed_delay=0):

        assert (x_lim is None) == (y_lim is None), "For now it is not possible to have only one of the two axes limited"
        assert in_loc.shape == (size_in, 2)
        assert out_loc.shape == (size_out, 2)

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
        self.out_loc = out_loc
        self.autapse = autapse
        self.cluster_decay = decay
        self.fixed_delay = fixed_delay

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

        r_variance_x = [0, start_width / 2]
        r_variance_y = [0, start_height / 2]

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
        grid, n_type, clusters = get_gaussian_mixture_config(N - size_in - size_out, mix, mu, inhibitory, Sigma,
                                                             x_range=x_lim, y_range=y_lim)

        # width = x_range[1] - x_range[0]
        # height = y_range[1] - y_range[0]

        for l in in_loc:
            mu_x_in, mu_y_in = l

            Sigma_in = 0 * (np.array(
                [
                    [start_width, 0],
                    [0, start_height]
                ])) ** 2
            Sigma_in = np.expand_dims(Sigma_in, 0)

            grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, 1, mu_x_in, mu_y_in,
                                                       Sigma=Sigma_in,
                                                       x_range=x_lim, y_range=y_lim)

        for l in out_loc:
            mu_x_out, mu_y_out = l

            Sigma_in = 0 * (np.array(
                [
                    [start_width, 0],
                    [0, start_height]
                ])) ** 2
            Sigma_in = np.expand_dims(Sigma_in, 0)

            grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, 1, mu_x_out, mu_y_out,
                                                       Sigma=Sigma_in,
                                                       x_range=x_lim, y_range=y_lim)
        input_index = np.array(range(size_out, size_out + size_in))

        output_index = np.array(range(0, size_out))
        # output_index = np.random.choice(range(size_in, N), size=(size_out, 1), replace=False)

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
            'out_loc': self.out_loc,
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
                 y0_mean, y0_scaling, size_in, size_out, in_loc, param_ranges, fixed_params,
                 act_func=sigmoid_activation, autapse=False, dt=0.0001,
                 x_lim=None, y_lim=None, propagation_vel=30, theta_window=None):

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
        self.autapse = autapse
        self.cluster_decay = decay
        self.fixed = fixed_params

        self.ranges = param_ranges
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
                         theta_window=theta_window, lr=lr, theta_y0=y0,
                         propagation_vel=propagation_vel)

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
            'dt': self.dt,
            'param_ranges': self.ranges,
            'fixed_params': self.fixed,
            'propagation_vel': self.propagation_vel
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


class GMMPopulationOld(DistDelayNetworkOld):
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
        self.autapse = autapse
        x_range = x_range
        y_range = y_range
        self.cluster_decay = decay
        self.fixed_delay = fixed_delay

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

class AdaptiveFlexiblePopulation(DistDelayNetwork):
    def __init__(self, N, x_range, y_range, dt, in_loc, out_loc, size_in, size_out, p_dict,
                 act_func=sigmoid_activation):
        assert p_dict.keys() == {'mix', 'mu_x', 'mu_y', 'variance_x', 'variance_y', 'correlation', 'inhibitory',
                                 'connectivity', 'weight_mean', 'weight_scaling', 'bias_mean', 'bias_scaling',
                                 'decay_mean', 'decay_scaling', 'in_scaling', 'in_mean', 'in_connectivity', 'lr_mean',
                                 'lr_scaling', 'theta0_mean', 'theta0_scaling', 'in_lr_mean', 'in_lr_scaling',
                                 'out_lr_mean', 'out_lr_scaling', 'out_theta0', 'out_mean',
                                 'out_scaling', 'out_connectivity'}

        self.init_evo_info(p_dict)
        self.N = N
        self.x_range = x_range
        self.y_range = y_range
        self.dt = dt
        self.in_loc = in_loc
        self.out_loc = out_loc
        self.size_in = size_in
        self.size_out = size_out

        self.mix = np.clip(p_dict['mix']['val'], p_dict['mix']['lims'][0], p_dict['mix']['lims'][1])
        if np.sum(self.mix) > 0:
            self.mix = self.mix / (np.sum(self.mix))
        else:
            self.mix = np.ones_like(self.mix) / len(self.mix)
        self.K = len(self.mix)

        self.mu_x = p_per_cluster(p_dict['mu_x']['val'], self.K, 1)
        self.mu_y = p_per_cluster(p_dict['mu_y']['val'], self.K, 1)
        self.inhibitory = p_per_cluster(p_dict['inhibitory']['val'], self.K, 1)

        self.weight_mean = p_per_cluster(p_dict['weight_mean']['val'], self.K, 2)
        self.weight_scaling = p_per_cluster(p_dict['weight_scaling']['val'], self.K, 2)
        self.weight_lims = p_dict['weight_mean']['lims']
        self.res_connectivity = p_per_cluster(p_dict['connectivity']['val'], self.K, 2)
        self.in_mean = p_per_cluster(p_dict['in_mean']['val'], self.K, 1)
        self.in_scaling = p_per_cluster(p_dict['in_scaling']['val'], self.K, 1)
        self.in_connectivity = p_per_cluster(p_dict['in_connectivity']['val'], self.K, 1)

        self.bias_mean = p_per_cluster(p_dict['bias_mean']['val'], self.K, 1)
        self.bias_scaling = p_per_cluster(p_dict['bias_scaling']['val'], self.K, 1)
        self.bias_lims = p_dict['bias_mean']['lims']

        self.decay_means = p_per_cluster(p_dict['decay_mean']['val'], self.K, 1)
        self.decay_scaling = p_per_cluster(p_dict['decay_scaling']['val'], self.K, 1)
        self.lr_mean = p_per_cluster(p_dict['lr_mean']['val'], self.K, 2)
        self.lr_mean = p_per_cluster(p_dict['lr_scaling']['val'], self.K, 2)
        self.theta0_mean = p_per_cluster(p_dict['theta0_mean']['val'], self.K, 1)
        self.theta0_scaling = p_per_cluster(p_dict['theta0_scaling']['val'], self.K, 1)
        self.in_lr_mean = p_per_cluster(p_dict['in_lr_mean']['val'], self.K, 1)
        self.out_lr_mean = p_per_cluster(p_dict['out_lr_mean']['val'], self.K, 1)
        self.in_lr_scaling = p_per_cluster(p_dict['in_lr_scaling']['val'], self.K, 1)
        self.out_lr_scaling = p_per_cluster(p_dict['out_lr_scaling']['val'], self.K, 1)
        self.out_theta0 = p_dict['out_theta0']['val']
        self.out_mean = p_per_cluster(p_dict['out_mean']['val'], self.K, 1)
        self.out_scaling = p_per_cluster(p_dict['out_scaling']['val'], self.K, 1)
        self.out_connectivity = p_per_cluster(p_dict['out_connectivity']['val'], self.K, 1)

        self.p_dict = p_dict

        # Get neuron location grid
        Sigma = self.get_loc_covariance()
        mu = np.array([self.mu_x, self.mu_y]).T

        grid, n_type, clusters = get_gaussian_mixture_config(N - size_in, self.mix, mu, self.inhibitory, Sigma,
                                                             x_range, y_range)

        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        mu_x_in, mu_y_in = self.in_loc

        Sigma_in = 0.002 * (np.array(
            [
                [width, 0],
                [0, height]
            ])) ** 2
        Sigma_in = np.expand_dims(Sigma_in, 0)

        grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, size_in, mu_x_in, mu_y_in,
                                                   Sigma=Sigma_in,
                                                   x_range=x_range, y_range=y_range)
        sort_key = np.argsort(clusters)
        clusters_sorted = clusters[sort_key]
        n_type_sorted = n_type[sort_key]
        grid_sorted = grid[sort_key]

        input_index = np.array(range(N - size_in, N))
        output_index = np.array(range(0, N - size_in))

        self.clusters = clusters_sorted

        # Get weight matrix
        W, bias, decay = self.get_cluster_based_params()

        super().__init__(weights=W, bias=bias, n_type=n_type_sorted, coordinates=grid_sorted, decay=decay,
                         input_n=input_index,
                         output_n=output_index, activation_func=act_func, dt=dt)

    def init_evo_info(self, p_dict):
        self.total_serial_p_size = 0
        for p_name in p_dict:
            assert p_dict[p_name].keys() == {'val', 'evolve', 'range', 'lims'}
            assert type(p_dict[p_name]['val']) == np.ndarray
            if p_dict[p_name]['evolve']:
                serial_size = p_dict[p_name]['val'].size
                self.total_serial_p_size += serial_size

class FlexiblePopulation(DistDelayNetworkOld):
    def __init__(self, N, x_range, y_range, dt, in_loc, size_in, size_out,
                 p_dict, act_func=sigmoid_activation):
        assert p_dict.keys() == {'mix', 'mu_x', 'mu_y', 'variance_x', 'variance_y', 'correlation', 'inhibitory',
                                 'connectivity', 'weight_mean', 'weight_scaling', 'bias_mean', 'bias_scaling',
                                 'decay_mean', 'decay_scaling', 'in_scaling', 'in_mean','in_connectivity'}

        self.init_evo_info(p_dict)
        self.N = N
        self.x_range = x_range
        self.y_range = y_range
        self.dt = dt
        self.in_loc = in_loc
        self.size_in = size_in
        self.size_out = size_out

        self.mix = np.clip(p_dict['mix']['val'], p_dict['mix']['lims'][0], p_dict['mix']['lims'][1])
        if np.sum(self.mix) > 0:
            self.mix = self.mix/(np.sum(self.mix))
        else:
            self.mix = np.ones_like(self.mix)/len(self.mix)
        self.K = len(self.mix)

        self.mu_x = p_per_cluster(p_dict['mu_x']['val'], self.K, 1)
        self.mu_y = p_per_cluster(p_dict['mu_y']['val'], self.K, 1)
        self.inhibitory = p_per_cluster(p_dict['inhibitory']['val'], self.K, 1)

        self.weight_mean = p_per_cluster(p_dict['weight_mean']['val'], self.K, 2)
        self.weight_scaling = p_per_cluster(p_dict['weight_scaling']['val'], self.K, 2)
        self.weight_lims = p_dict['weight_mean']['lims']
        self.res_connectivity = p_per_cluster(p_dict['connectivity']['val'], self.K, 2)
        self.in_mean = p_per_cluster(p_dict['in_mean']['val'], self.K, 1)
        self.in_scaling = p_per_cluster(p_dict['in_scaling']['val'], self.K, 1)
        self.in_connectivity = p_per_cluster(p_dict['in_connectivity']['val'], self.K, 1)

        self.bias_mean = p_per_cluster(p_dict['bias_mean']['val'], self.K, 1)
        self.bias_scaling = p_per_cluster(p_dict['bias_scaling']['val'], self.K, 1)
        self.bias_lims = p_dict['bias_mean']['lims']

        self.decay_means = p_per_cluster(p_dict['decay_mean']['val'], self.K, 1)
        self.decay_scaling = p_per_cluster(p_dict['decay_scaling']['val'], self.K, 1)
        self.p_dict = p_dict


        # Get neuron location grid
        Sigma = self.get_loc_covariance()
        mu = np.array([self.mu_x, self.mu_y]).T

        grid, n_type, clusters = get_gaussian_mixture_config(N - size_in, self.mix , mu, self.inhibitory, Sigma,
                                                             x_range, y_range)

        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        mu_x_in, mu_y_in = self.in_loc

        Sigma_in = 0.002 * (np.array(
            [
                [width, 0],
                [0, height]
            ])) ** 2
        Sigma_in = np.expand_dims(Sigma_in, 0)

        grid, n_type, clusters = set_inout_cluster(grid, n_type, clusters, size_in, mu_x_in, mu_y_in,
                                                   Sigma=Sigma_in,
                                                   x_range=x_range, y_range=y_range)
        sort_key = np.argsort(clusters)
        clusters_sorted = clusters[sort_key]
        n_type_sorted = n_type[sort_key]
        grid_sorted = grid[sort_key]

        input_index = np.array(range(N - size_in, N))
        output_index = np.array(range(0, N - size_in))

        self.clusters = clusters_sorted

        # Get weight matrix
        W, bias, decay = self.get_cluster_based_params()

        super().__init__(weights=W, bias=bias, n_type=n_type_sorted, coordinates=grid_sorted, decay=decay, input_n=input_index,
                         output_n=output_index, activation_func=act_func, dt=dt)


    def get_serialized_parameters(self):
        def scale_down(params, middle, scale):
            scaled = params - middle
            scaled = scaled / scale
            return scaled
        serialized = []

        for p_name in self.p_dict:
            if self.p_dict[p_name]['evolve']:
                val = self.p_dict[p_name]['val']
                range = self.p_dict[p_name]['range']
                middle = np.sum(range) / 2
                scale = range[1] - range[0]
                flat_val = np.reshape(val, (np.prod(val.shape),))
                scaled_flat_val = scale_down(flat_val, middle, scale)
                serialized += list(scaled_flat_val)
        assert len(serialized) == self.total_serial_p_size
        return serialized

    def get_p_dict_from_serialized(self, serialized_parameters):
        assert len(serialized_parameters) == self.total_serial_p_size
        new_parameter_dict = {}
        for p_name in self.p_dict:
            if self.p_dict[p_name]['evolve']:
                p_shape = self.p_dict[p_name]['val'].shape
                p_lims = self.p_dict[p_name]['lims']
                p_range = self.p_dict[p_name]['range']

                p_len = np.prod(p_shape)
                p_middle = np.sum(p_range) / 2
                p_scale = p_range[1] - p_range[0]

                new_p_flat_scaled = np.array(serialized_parameters[:p_len])
                new_p_flat_unclipped = scale_up(new_p_flat_scaled, p_middle, p_scale)
                new_p_flat = np.clip(new_p_flat_unclipped, p_lims[0], p_lims[1])
                new_p = np.reshape(new_p_flat, p_shape)

                serialized_parameters = serialized_parameters[p_len:]

                new_p_dict = {}
                new_p_dict['val'] = new_p
                new_p_dict['evolve'] = True
                new_p_dict['range'] = p_range
                new_p_dict['lims'] = p_lims
                new_parameter_dict[p_name] = new_p_dict
            else:
                new_parameter_dict[p_name] = self.p_dict[p_name]

        assert len(serialized_parameters) == 0
        return new_parameter_dict

    def get_new_network_from_serialized(self, serialized_parameters):
        new_p_dict = self.get_p_dict_from_serialized(serialized_parameters)
        net_args = {
            'N': self.N,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'size_in': self.size_in,
            'size_out': self.size_out,
            'dt': self.dt,
            'in_loc': self.in_loc,
            'size_in': self.size_in,
            'size_out': self.size_out,
            'act_func': self.activation_func,
            'p_dict': new_p_dict
        }
        new_flex_pop = FlexiblePopulation(**net_args)
        return new_flex_pop

    def init_evo_info(self, p_dict):
        self.total_serial_p_size = 0
        for p_name in p_dict:
            assert p_dict[p_name].keys() == {'val', 'evolve', 'range', 'lims'}
            assert type(p_dict[p_name]['val']) == np.ndarray
            if p_dict[p_name]['evolve']:
                serial_size = p_dict[p_name]['val'].size
                self.total_serial_p_size += serial_size

    def get_cluster_based_params(self, dist='normal'):
        assert dist in ['normal', 'uniform']
        N = self.N
        K = self.K
        W = np.zeros((N, N))
        b = np.zeros((N,))
        decay = np.zeros((N,))

        means = np.zeros(shape=(K + 1, K + 1))
        scales = np.zeros(shape=(K + 1, K + 1))
        b_means = np.zeros(shape=(K + 1,))
        b_scales = np.zeros(shape=(K + 1,))
        connectivity = np.zeros(shape=(K + 1, K + 1))
        decay_means = np.zeros(shape=(K + 1,))
        decay_scales = np.zeros(shape=(K + 1,))

        means[:-1, :-1] = self.weight_mean
        means[:-1, -1] = self.in_mean
        b_means[:-1] = self.bias_mean
        b_means[-1] = 0
        scales[:-1, :-1] = self.weight_scaling
        scales[:-1, -1] = self.in_scaling
        b_scales[:-1] = self.bias_scaling
        b_scales[-1] = 0
        connectivity[:-1, :-1] = self.res_connectivity
        connectivity[:-1, -1] = self.in_connectivity
        decay_means[:-1] = self.decay_means
        decay_means[-1] = 1
        decay_scales[:-1] = self.decay_scaling
        decay_scales[-1] = 1

        for c1 in range(K + 1):
            c_inds_1 = np.argwhere(self.clusters == c1)
            c_size_1 = len(c_inds_1)
            if c_size_1 > 0:
                b_mean = b_means[c1]
                b_scale = b_scales[c1]
                d_mean = decay_means[c1]
                d_scale = decay_scales[c1]
                if dist == 'normal':
                    b_vec = np.random.normal(loc=b_mean, scale=b_scale, size=(c_size_1,))
                    a_vec = np.random.normal(loc=d_mean, scale=d_scale, size=(c_size_1,))
                if dist == 'uniform':
                    b_vec = np.random.uniform(low=b_mean-.5 * b_scale, high=b_mean+.5 * b_scale, size=(c_size_1,))
                    a_vec = np.random.uniform(low=d_mean-.5 * d_scale, high=d_mean+.5 * d_scale, size=(c_size_1,))
                b_vec = np.clip(b_vec, self.bias_lims[0], self.bias_lims[1])
                a_vec = np.clip(a_vec, 0, 1)
                b[np.min(c_inds_1): np.max(c_inds_1) + 1] = b_vec
                decay[np.min(c_inds_1): np.max(c_inds_1) + 1] = a_vec

                for c2 in range(K + 1):
                    clip_lims = self.weight_lims
                    if c2 == K:
                        clip_lims = self.p_dict['in_mean']['lims']

                    c_inds_2 = np.argwhere(self.clusters == c2)
                    c_size_2 = len(c_inds_2)
                    if c_size_2 > 0:
                        if c1 == K:
                            c_mat = np.zeros(shape=(c_size_1, c_size_2))
                        else:
                            mean = means[c1, c2]
                            scale = scales[c1, c2]

                            if dist == 'normal':
                                c_mat = np.random.normal(loc=mean, scale=scale, size=(c_size_1, c_size_2))
                            if dist == 'uniform':
                                c_mat = np.random.uniform(low=mean-.5*scale, high=mean+.5*scale, size=(c_size_1, c_size_2))
                            c_mat = np.clip(c_mat, clip_lims[0], clip_lims[1])
                        conn = connectivity[c1, c2]
                        connectivity_mask = np.random.uniform(0, 1, size=(c_size_1, c_size_2)) < conn
                        c_mat *= connectivity_mask # Prune according to connectivity parameter
                        W[np.min(c_inds_1):np.max(c_inds_1)+1, np.min(c_inds_2):np.max(c_inds_2)+1] = c_mat
        return W, b, decay


    def get_loc_covariance(self):
        var_mat = np.zeros((self.K, 2, 2))
        corr_mat = np.ones((self.K, 2, 2))
        x_var = p_per_cluster(self.p_dict['variance_x']['val'], self.K, 1)
        y_var = p_per_cluster(self.p_dict['variance_y']['val'], self.K, 1)
        xy_corr = p_per_cluster(self.p_dict['correlation']['val'], self.K, 1)
        for i in range(self.K):
            var_mat[i, 0, 0] = x_var[i]
            var_mat[i, 1, 1] = y_var[i]
            corr_mat[i, 0, 1] = xy_corr[i]
            corr_mat[i, 1, 0] = xy_corr[i]

        Sigma = np.matmul(np.matmul(var_mat, corr_mat), var_mat)
        return Sigma

class ESHyperNEATPopulation(DistDelayNetwork):

    def __init__(self, ESNet, activation_function=None, dt=.0005):

        # input_coordinates = ESNet.substrate.input_coordinates
        # reservoir_coordinates = ESNet.es_hyperneat()[0]
        # pheno = ESNet.create_phenotype_network()
        # n_neurons = len(pheno.node_evals) + len(pheno.input_nodes)
        # res_weights_mat = np.zeros(shape=(n_neurons, n_neurons))
        # for node, _, _, _, _, links in pheno.node_evals:
        #     for i, w in links:
        #         node_input, output = node, i
        #         res_weights_mat[node_input, output] = w
        if activation_function is None:
            activation_function = sigmoid_activation
        ddn_phenotype = self.create_DDN_phenotype(ESNet, activation_function, dt)
        super().__init__(**ddn_phenotype)

    def create_DDN_phenotype(self, ESNet, activation_function, dt):
        input_coordinates = ESNet.substrate.input_coordinates
        output_coordinates = ESNet.substrate.output_coordinates

        input_nodes = list(range(len(input_coordinates)))
        output_nodes = list(range(len(input_nodes), len(
            input_nodes) + len(output_coordinates)))
        hidden_idx = len(input_coordinates) + len(output_coordinates)

        coordinates, indices, draw_connections, node_evals = [], [], [], []
        nodes = {}

        coordinates.extend(input_coordinates)
        coordinates.extend(output_coordinates)
        indices.extend(input_nodes)
        indices.extend(output_nodes)

        # Map input and output coordinates to their IDs.
        coords_to_id = dict(zip(coordinates, indices))

        # Where the magic happens.
        hidden_nodes, connections = ESNet.es_hyperneat()

        # Map hidden coordinates to their IDs.
        for x, y in hidden_nodes:
            coords_to_id[x, y] = hidden_idx
            hidden_idx += 1

        # For every coordinate:
        # Check the connections and create a node with corresponding connections if appropriate.
        for (x, y), idx in coords_to_id.items():
            for c in connections:
                if c.x2 == x and c.y2 == y:
                    draw_connections.append(c)
                    if idx in nodes:
                        initial = nodes[idx]
                        initial.append((coords_to_id[c.x1, c.y1], c.weight))
                        nodes[idx] = initial
                    else:
                        nodes[idx] = [(coords_to_id[c.x1, c.y1], c.weight)]

        # Combine the indices with the connections/links;
        # forming node_evals used by the RecurrentNetwork.
        for idx, links in nodes.items():
            node_evals.append((idx, ESNet.activation, sum, 0.0, 1.0, links))

        n_neurons = len(nodes) + len(input_nodes) + len(output_nodes)
        ddn_weights_mat = np.zeros(shape=(n_neurons, n_neurons))
        ddn_bias = np.zeros(shape=(n_neurons,))
        n_type = np.ones_like(ddn_bias)

        for i in nodes:
            for j, w in nodes[i]:
                ddn_weights_mat[i, j] = w

        for n in node_evals:
            ddn_bias[n[0]] = n[3]
            n_type[n[0]] = n[4]

        ddn_coordinates = np.zeros(shape=(n_neurons, 2))

        for coords in coords_to_id:
            id = coords_to_id[coords]
            ddn_coordinates[id, :] = coords

        ddn_decay = np.ones_like(ddn_bias) * .95
        # Visualize the network?
        # if filename is not None:
        # draw_es(coords_to_id, draw_connections, "pureples_testing/test_file.png")

        pheno_dict = {
            "weights": ddn_weights_mat,
            "bias": ddn_bias,
            "n_type": n_type,
            "coordinates": ddn_coordinates,
            "decay": ddn_decay,
            "input_n": input_nodes,
            "output_n": list(nodes.keys()),
            "activation_func": activation_function,
            "dt": dt
        }

        return pheno_dict


def scale_up(params, middle, scale):
    scaled = params * scale
    scaled = scaled + middle
    return scaled


# static functions
def p_per_cluster(p_config, K, dims):
    if len(p_config)>1:
        assert len(p_config.shape) == dims
        assert p_config.shape == tuple([K] * dims)
        return p_config
    elif len(p_config) == 1:
        p = np.ones(shape=tuple([K] * dims)) * p_config
        return p


def get_empty_parameters(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    start_mix = np.ones((k,)) / k
    start_mu = np.zeros((k, 2))
    start_var = np.zeros((k, 2))
    start_corr = np.zeros((k,))
    inhib_start = np.zeros((k,))
    conn_start = np.zeros((k + 1, k + 1))
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
        'size_out': N - insize,
        'in_loc': in_loc,
        'act_func': activation_function,
        'dt': dt,
        'buffersize': buffersize
    }
    return net_params


def get_empty_parameters_old(N, insize, k, dt, buffersize, activation_function, x_range, y_range, in_loc):
    start_mix = np.ones((k,)) / k
    start_mu = np.zeros((k, 2))
    start_var = np.zeros((k, 2))
    start_corr = np.zeros((k,))
    inhib_start = np.zeros((k,))
    conn_start = np.zeros((k + 1, k + 1))
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
        'size_out': N - insize,
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
    lr_mean = np.ones((k + 1, k + 1)) * .1
    lr_scaling = np.ones((k + 1, k + 1)) * .1
    y0_mean = np.ones((k + 1,))
    y0_scaling = np.ones((k + 1,)) * .1
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
    cluster_inds = list(range(int(np.max(clusters)) + 1, int(np.max(clusters)) + N + 1))
    new_clusters[:N] = cluster_inds  #np.max(clusters) + 1
    new_clusters[N:] = clusters
    return new_grid, new_ntype, new_clusters


def clustered_pop_pruning(W, connectivity, clusters, cluster_connectivity=None):
    if cluster_connectivity is None:
        cluster_connectivity = np.ones_like(clusters)
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

