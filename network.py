import numpy as np
import config


class DistDelayNetwork(object):
    """
        Class for distance based delay networks.
    """

    # TODO: make new branch and take out n_type
    def __init__(self, weights, bias, n_type, coordinates, decay,
                 input_n=np.array([0, 1, 2]), output_n=np.array([-3, -2, -1]),
                 activation_func=None, dt=0.0005, buffersize=100, theta_window=None, theta_y0=None,
                 x_range=(0, .002), y_range=(0, .004),
                 lr=1, var_delays=True):
        """
        TODO: update doc
        Constructor for distance based delay networks
        :param weights: ndarray
            N by N array, with N the number of neurons. This array contains the connection weights
        :param bias: ndarray
            N by 1 array, with N the number of neurons. This array contains the bias weights
        :param n_type: ndarray
            N by 1 array, with N the number of neurons. This array contains the neuron type (1 for excitatory, -1 for
            inhibitory.
        :param coordinates: ndarray
            N by dim array, with N the number of neurons and dim the number of spatial dimensions. This array
            stores the spatial coordinates (in metres) of the neurons in the network. Distance matrix is computed based
            on this.
        :param input_n: ndarray
            Array with indices of input neurons
        :param output_n: ndarray
            Array with indices of output neurons
        :param activation_func: function
            Neuron activation function
        :param dt: float
            The physical time that one simulation step represents, in seconds.
        :param buffersize: int
            Size of buffer array, should be larger than buffer margin.
        :param decay: float
            Decay parameter for neuron input.
        """
        # compute continuous distance matrix
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]
        max_dist = np.sqrt(width ** 2 + height ** 2)
        self.spatial_dist_continuous = coordinates2distance(coordinates)
        longest_delay_possible = buffersize * dt
        if buffersize == 1:
            longest_delay_possible = 1
        longest_delay_needed = (max_dist) / config.propagation_vel
        assert longest_delay_needed <= longest_delay_possible, 'Buffer not large enough'

        self.coordinates = coordinates
        self.dt = dt
        self.N = coordinates.shape[0]  # Number of neurons
        self.W = weights
        self.WBias = bias
        self.n_type = n_type
        self.buffersize = buffersize
        # self.X_init = np.repeat(np.reshape(bias, (len(bias), 1)), self.buffersize, 1)
        # self.X = self.X_init
        self.A_init = np.zeros((self.N, buffersize))
        self.A = np.copy(self.A_init)

        self.neurons_in = input_n  # Indices for input neurons
        self.neurons_out = output_n  # Indices for output neurons
        self.decay = decay
        self.weight_decay = .0001
        self.x_range = x_range
        self.y_range = y_range
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

        if var_delays:
            # Discretized distance matrix according to given dt
            dist_per_step = dt * config.propagation_vel
            self.D = np.asarray(np.ceil(self.spatial_dist_continuous / dist_per_step), dtype='int32')
        else:
            self.D = np.ones_like(self.spatial_dist_continuous)
            np.fill_diagonal(self.D, 0)
        # Compute masked weight matrices
        self.W_masked_list = []
        self.lr_masked_list = []
        if not (self.W is None) and var_delays:
            self.compute_masked_W()
            self.compute_masked_lr()

        self.var_delays = var_delays

    def reset_network(self):
        """
        Resets network activity to initial state.
        :return: None
        """
        self.A = self.A_init

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

        if self.buffersize > 1 and self.var_delays:
            # Compute input to reservoir neurons on next time step taking delays into consideration
            neuron_inputs = np.copy(self.WBias)  # Add bias weights first
            for d, masked_weights in enumerate(self.W_masked_list):
                # Keep adding activations from d time steps ago
                # multiplied with weights of length d (as formalized in paper)
                neuron_inputs += np.matmul(masked_weights, self.A[:, d] * self.n_type)
        else:
            neuron_inputs = np.matmul(self.W, self.A[:, 0] * self.n_type) + self.WBias

        # apply activation function
        y = self.activation_func(neuron_inputs)
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
        dW = dW - self.weight_decay * dW
        # dW = dW / np.repeat(np.expand_dims(self.theta, -1), act_pre.shape, axis=-1).T
        # dWd = dWd * np.repeat(np.expand_dims(sigmoid_der(act_post), -1), act_pre.shape, axis=-1).T
        dW = dW.T * self.lr
        self.W += dW

    def delayed_BCM(self):
        assert self.buffersize > 2, 'Need a non-zero delay for delayed BCM'
        self.update_theta()
        act_post = self.A[:, 0]
        post_term = np.expand_dims(act_post * (act_post - self.theta), -1)
        for d, Wd in enumerate(self.W_masked_list):
            act_pre = self.A[:, d]
            dWd = (post_term @ np.expand_dims(act_pre, 0)).T
            dWd = dWd - self.weight_decay * dWd
            # dWd = dWd / np.repeat(np.expand_dims(self.theta, -1), act_pre.shape, axis=-1).T
            # dWd = dWd * np.repeat(np.expand_dims(sigmoid_der(act_post), -1), act_pre.shape, axis=-1).T
            dWd = dWd.T * self.lr_masked_list[d]
            Wd += dWd

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
                                   activation_func=self.activation_func, dt=new_dt, buffersize=new_buffersize,
                                   x_range=self.x_range, y_range=self.y_range)
        return new_net


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
