import numpy as np
import config


class DistDelayNetwork(object):
    """
        Class for distance based delay networks.
    """

    # TODO: make new branch and take out n_type
    def __init__(self, weights, bias, n_type, coordinates, decay,
                 input_n=np.array([0, 1, 2]), output_n=np.array([-3, -2, -1]),
                 activation_func=None, dt=0.0005, theta_window=None, theta_y0=None,
                 lr=1, propagation_vel = 30):

        self.x_range = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
        self.y_range = (np.min(coordinates[:, 1]), np.max(coordinates[:, 1]))
        self.spatial_dist_continuous = coordinates2distance(coordinates)

        # Discretized distance matrix according to given dt
        dist_per_step = dt * propagation_vel
        self.propagation_vel = propagation_vel
        self.D = np.asarray(np.ceil(self.spatial_dist_continuous / dist_per_step), dtype='int32')

        longest_delay_needed = np.max(self.D) + 1

        self.coordinates = coordinates
        self.dt = dt
        self.N = coordinates.shape[0]  # Number of neurons
        self.W = weights
        self.WBias = bias
        self.n_type = n_type
        self.B = longest_delay_needed  # Buffer size
        self.var_delays = self.B > 2
        self.A_init = np.zeros((self.N, self.B))
        self.A = np.copy(self.A_init)
        self.neuron_inputs = np.zeros((self.N,))

        self.neurons_in = input_n  # Indices for input neurons
        self.neurons_out = output_n  # Indices for output neurons
        self.decay = decay
        self.weight_decay = 0.01

        self.lr = lr * np.array(self.W > 0, dtype='uint8')
        self.theta = np.ones((self.N,))
        self.theta_window = theta_window
        self.theta_y0 = theta_y0
        if self.theta_window is None:
            self.theta_window = self.A.shape[1]
        assert self.theta_window <= self.A.shape[1], 'Window size for theta can not be larger than buffer size for ' \
                                                     'net activity. '

        if self.theta_y0 is None:
            self.theta_y0 = 1

        if activation_func is None:
            self.activation_func = tanh_activation
        else:
            self.activation_func = activation_func

        self.mid_dist = (np.max(self.D) - 1) / 2

        # Compute masked weight matrices
        self.W_masked_list = [self.W]
        self.lr_masked_list = [self.lr]
        if not (self.W is None) and self.var_delays:
            self.compute_masked_W()
            self.compute_masked_lr()

        self.W_masked_list_init = [np.copy(partial_W) for partial_W in self.W_masked_list]

        # compute connectivity matrix for use in structural plasticity
        self.adjacency = np.absolute(self.W) > 0

    def reset_network(self):
        """
        Resets network activity to initial state.
        :return: None
        """
        self.reset_activity()
        self.reset_weights()

    def reset_activity(self):
        self.A = np.copy(self.A_init)

    def reset_weights(self):
        self.W_masked_list = [np.copy(partial_W) for partial_W in self.W_masked_list_init]

    def compute_masked_W(self):
        """
        Creates a list of masked weight matrices, i.e. weight matrices containing only the weights of connections with
        a specified delay.
        Returns: None
        """
        self.W_masked_list.clear()
        for buffStep in range(np.max(self.D) + 1):
            # Create mask for each buffer step
            mask = self.D == buffStep
            # Elementwise product with buffer mask to only add activity to correct buffer step
            buffW = np.multiply(mask, self.W)
            self.W_masked_list.append(buffW)

    def compute_masked_lr(self):
        self.lr_masked_list.clear()
        excitatory_pre = np.repeat(np.expand_dims(np.array(self.n_type > 0, dtype='uint8'), 0), self.N, axis=0)
        for buffStep in range(np.max(self.D) + 1):
            # fix zero weights
            buffLr = self.lr * np.array(self.W_masked_list[buffStep] > 0, dtype='uint8')
            # only update weights with excitatory presynaptic units
            buffLr = buffLr * excitatory_pre
            self.lr_masked_list.append(buffLr)

    def update_step(self, input):
        # Shift buffers in time appropriately
        if self.A.shape[1] > 1:
            self.A[:, 1:] = self.A[:, :-1]  # shifts from present to past

        if self.B > 1 and self.var_delays:
            # Compute input to reservoir neurons on next time step taking delays into consideration
            self.neuron_inputs = np.copy(self.WBias)  # Add bias weights first
            for d, masked_weights in enumerate(self.W_masked_list):
                # Keep adding activations from d time steps ago
                # multiplied with weights of length d (as formalized in paper)
                self.neuron_inputs += np.matmul(masked_weights, self.A[:, d] * self.n_type)
        else:
            self.neuron_inputs = np.matmul(self.W, self.A[:, 0] * self.n_type) + self.WBias

        # apply activation function
        y = self.activation_func(self.neuron_inputs)
        self.A[:, 0] = (1 - self.decay) * self.A[:, 0] + self.decay * y

        # Input neuron is forced to input value
        self.clamp_input(input)

    def update_step_adaptive(self, input):
        self.update_step(input)
        if self.var_delays:
            self.delayed_BCM()
        else:
            self.simple_BCM()

    def clamp_input(self, input_array):
        """
        Set the input value of the input neurons to that of a given input array.
        :param input_array: ndarray
            N_i by 1 array with N_i the number of input neurons.
        :return: None
        """
        assert len(input_array) == len(self.neurons_in)
        input_ind = self.neurons_in
        input_ind = np.reshape(input_ind, (len(input_ind),))
        self.A[input_ind, 0] = input_array

    def simple_BCM(self):
        assert self.theta_window > 1, 'Need a activity history to compute theta'
        self.update_theta()
        act_post = self.A[:, 0]
        post_term = np.expand_dims(act_post * (act_post - self.theta), -1)
        act_pre = self.A[:, 0]
        dW = (post_term @ np.expand_dims(act_pre, 0)).T
        dW = dW - self.weight_decay * self.W
        # dW = dW / np.repeat(np.expand_dims(self.theta, -1), act_pre.shape, axis=-1).T
        # dWd = dWd * np.repeat(np.expand_dims(sigmoid_der(act_post), -1), act_pre.shape, axis=-1).T
        dW = dW.T * self.lr
        self.W += dW

    def delayed_BCM(self):
        assert self.B >= 2, 'Need a non-zero delay for delayed BCM'
        self.update_theta()
        act_post = self.A[:, 0]
        post_term = np.expand_dims(act_post * (act_post - self.theta), -1)
        # dW = np.zeros_like(self.W)
        for d, Wd in enumerate(self.W_masked_list):
            act_pre = self.A[:, d]
            dWd = (post_term @ np.expand_dims(act_pre, 0)).T
            dWd = dWd - self.weight_decay * Wd
            # dWd = dWd / np.repeat(np.expand_dims(self.theta, -1), act_pre.shape, axis=-1).T
            # dWd = dWd * np.repeat(np.expand_dims(sigmoid_der(act_post), -1), act_pre.shape, axis=-1).T
            dWd = dWd.T * self.lr_masked_list[d]
            Wd += dWd
            # dW += dWd
            # Structural plasticity
            # Wd *= self.adjacency

        # self.grow(dW)
        # self.prune()

    def grow(self, dW):
        thresholds_d_grow = 10
        scaled_distances_grow = 1 * ((np.max(self.D) - self.D) - (thresholds_d_grow + self.mid_dist))
        p_grow = sigmoid_activation(.5*np.abs(dW)-5) * sigmoid_activation(scaled_distances_grow)
        rng = np.random.uniform(size=p_grow.shape)
        grow = rng < p_grow
        grow = grow * (self.adjacency == 0)
        self.adjacency = self.adjacency + np.array(grow, dtype='int8')

    def prune(self, thresholds_w=.1, thresholds_d=4):
        current_weights = sum(self.W_masked_list)
        absolute_weights = np.absolute(current_weights)
        mid_weight = .5
        scaled_absolute_weights = (-absolute_weights + (thresholds_w + mid_weight))
        scaled_distances_prune = .5 * (self.D - (thresholds_d + self.mid_dist))
        p_prune = sigmoid_activation(scaled_distances_prune) * sigmoid_activation(-scaled_absolute_weights)
        rng = np.random.uniform(size=p_prune.shape)
        prune = rng < p_prune
        prune = prune * (self.adjacency > 0)
        self.adjacency = self.adjacency * (1 - np.array(prune, dtype='int8'))

    def update_theta(self):
        hist_mat = np.copy(self.A[:, :self.theta_window])
        hist_mat = np.average(hist_mat, axis=1)**2 / self.theta_y0
        self.theta = hist_mat

    def create_similar_by_max_delay(self, new_max_delay):

        # current_max_delay = np.max(self.D)

        width = self.x_range[1] - self.x_range[0]
        height = self.y_range[1] - self.y_range[0]
        max_possible_dist = np.sqrt(width ** 2 + height ** 2)
        max_current_dist = np.max(self.spatial_dist_continuous)
        fraction_of_max = max_current_dist / max_possible_dist

        ds = max_current_dist / new_max_delay  # distance discretized step
        new_dt = ds / config.propagation_vel

        new_buffersize = int(np.ceil(new_max_delay / fraction_of_max) + 1)

        # current_B = self.buffersize
        # corresponding_ds = max_possible_dist/(current_B)
        # corresponding_dt = corresponding_ds / config.propagation_vel

        # scaling = new_max_delay / current_max_delay
        # new_B = int(np.ceil(scaling * current_B) + 2)
        # new_dt = corresponding_dt / scaling
        new_net = DistDelayNetwork(weights=self.W, bias=self.WBias, n_type=self.n_type, coordinates=self.coordinates,
                                   decay=self.decay, input_n=self.neurons_in, output_n=self.neurons_out,
                                   activation_func=self.activation_func, dt=new_dt, buffersize=new_buffersize)
        return new_net

    def get_current_power_bio(self, activation_cost=.005, synapse_cost=.001, propagation_cost=.005):
        """
        Computes an estimate of the energy consumption/power of the network at the current timestep.
        Depends on activation, synaptic transmissions and distances.

        Args:
            activation_cost: float
                cost scalar for firing frequency
            synapse_cost: float
                cost scalar for synaptic transmissions
            propagation_cost:
                cost scalar for signal propagation per length unit

        Returns: float
            Current energy consumption
        """

        activation_energy_per_n = self.A[:, 0] * activation_cost
        synapse_energy_per_n = np.matmul(np.absolute(sum(self.W_masked_list)), self.A[:, 0]) * synapse_cost
        propagation_energy_per_n = np.matmul(self.D * self.dt * config.propagation_vel, self.A[:, 0]) * propagation_cost

        total_activation_energy = np.sum(activation_energy_per_n)
        total_synapse_energy = np.sum(synapse_energy_per_n)
        total_propagation_energy = np.sum(propagation_energy_per_n)

        total_energy = total_activation_energy + total_synapse_energy + total_propagation_energy
        return total_energy

# Static functions
def coordinates2distance(coordinates):
    """
    Transforms a spatial configuration of neurons to a distance (adjacency) matrix.
    :param coordinates: ndarray
        N by dims array with N the number of neurons and dims the number of spatial dimensions. Should contain the
        spatial coordinates in a 2D space of each neuron.
    :return: ndarray
        N by N array containing the spatial distance between each neuron.
    """
    N = coordinates.shape[0]

    D = np.zeros((N, N))

    def dist(dist_x, dist_y):
        return np.sqrt(dist_x ** 2 + dist_y ** 2)

    for i in range(N):
        for j in range(N):
            if not i == j:
                dist_x = np.abs(coordinates[i, 0] - coordinates[j, 0])
                dist_y = np.abs(coordinates[i, 1] - coordinates[j, 1])
                d = dist(dist_x, dist_y)
                D[i, j] = d
    return D


def stepwise_activation(neuron_input, threshold=0.0):
    """
    Performs threshold activation on a neuron input array
    :param neuron_input: ndarray
        N by 1 array with N number of neurons that encodes the input of all neurons.
    :param threshold: float
        Threshold value for all neurons
    :return: ndarray
        N by 1 float array with neuron activation.
    """
    x = neuron_input
    y = np.asarray(x > threshold, dtype='float64')
    return y


def sigmoid_activation(neuron_input):
    """
    Performs sigmoid activation on a neuron input array
    :param neuron_input: ndarray
        N by 1 array with N number of neurons that encodes the input of all neurons.
    :return: ndarray
        N by 1 float array with neuron activation.
    """
    x = neuron_input
    y = 1 / (1 + np.exp(-x))
    return y


def tanh_activation(neuron_input):
    x = neuron_input
    y = np.tanh(x)
    return y


def elu(neuron_input):
    e = np.e
    z = neuron_input
    if z >= 0:
        return z
    else:
        return (e ** z - 1)


def relu(neuron_input):
    return np.maximum(neuron_input, np.zeros_like(neuron_input))


def sigmoid_der(x):
    return sigmoid_activation(x) * (1-sigmoid_activation(x))


def get_multi_activation(activation_funcs, index_ranges):
    assert len(activation_funcs) == len(index_ranges), 'Number of slices should be equal to number of activation funcs'

    def multi_activation(neuron_input):
        activations = []
        for i, act_f in enumerate(activation_funcs):
            partial_in = neuron_input[index_ranges[i][0]:index_ranges[i][1]]
            if not act_f is None:
                partial_act = act_f(partial_in)
                activations.append(partial_act)
            else:
                activations.append(partial_in)
        activations = np.concatenate(activations)
        return activations

    return multi_activation
