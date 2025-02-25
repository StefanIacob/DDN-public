import numpy as np
import config


class DistDelayNetwork(object):
    """
        Class for distance based delay networks.
    """
    # TODO: make new branch and take out n_type
    def __init__(self, weights, bias, n_type, coordinates, decay,
                 input_n=np.array([0, 1, 2]), output_n=np.array([-3, -2, -1]),
                 activation_func=None, dt=0.0005, buffersize=100, x_range=(0, .002), y_range=(0, .004)):
        """
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
        longest_delay_possible = (buffersize - 1) * dt
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
        self.X_init = np.repeat(np.reshape(bias, (len(bias), 1)), self.buffersize, 1)
        self.X = self.X_init
        self.A_init = np.zeros((self.N,))
        self.A = self.A_init

        self.neurons_in = input_n  # Indices for input neurons
        self.neurons_out = output_n  # Indices for output neurons
        self.decay = decay
        self.x_range = x_range
        self.y_range = y_range

        if activation_func is None:
            self.activation_func = tanh_activation
        else:
            self.activation_func = activation_func

        # Discretized distance matrix according to given dt
        dist_per_step = dt * config.propagation_vel
        self.D = np.asarray(np.ceil(self.spatial_dist_continuous / dist_per_step), dtype='int32')

        # Compute masked weight matrices
        self.W_masked_list = []
        if not (self.W is None):
            self.compute_masked_W()

    def reset_network(self):
        """
        Resets network activity to initial state.
        :return: None
        """
        self.X = self.X_init
        self.A = self.A_init

    def compute_masked_W(self):
        for buffStep in range(self.buffersize):
            # Create mask for each buffer step
            mask = self.D == buffStep
            # Elementwise product with buffer mask to only add activity to correct buffer step
            buffW = np.multiply(mask, self.W)
            self.W_masked_list.append(buffW)

    def apply_time_delay(self):
        """
        Apply time delay on neuron output and add to neuron input at corresponding time point.
        :param neuron_output: ndarray
            N by 1 array with current neuron output (after nonlinear activation)
        :return:
            None
        """
        neuron_output = self.A
        for buffStep in range(self.buffersize):
            buffW = self.W_masked_list[buffStep]
            # Apply weight scaling
            x_new = np.matmul(buffW, neuron_output)
            self.X[:, buffStep] += x_new

    def update_step(self, input):
        """
        Performs a simulation step of the model.
        :return: None
        """
        # Shift buffers in time appropriately

        if self.buffersize > 1:
            self.X[:, :-1] = self.X[:, 1:]  # shifts from future to present
            # set initial future input to neuron bias
            self.X[:, -1] = self.WBias

        # Apply nonlinear activation function on current neuron input
        y = self.activation_func(self.X[:, 0])
        y = np.multiply(y, self.n_type)
        self.A = (1 - self.decay) * self.A + self.decay * y

        self.clamp_input(input)

        # Apply time delay and update neuron input
        if self.buffersize > 1:
            self.apply_time_delay()
        else:
            self.X[:, 0] = np.matmul(self.W, self.A)

    def apply_input(self, input_array):
        """
        Add the input array to the input neurons.
        :param input_array: ndarray
            N_i by 1 array with N_i the number of input neurons.
        :return: None
        """
        assert len(input_array) == len(self.neurons_in)
        input_ind = self.neurons_in
        input_ind = np.reshape(input_ind, (len(input_ind),))
        self.X[input_ind, 0] += input_array

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
        self.A[input_ind] = input_array

    def create_similar_by_max_delay(self, new_max_delay):

        # current_max_delay = np.max(self.D)

        width = self.x_range[1] - self.x_range[0]
        height = self.y_range[1] - self.y_range[0]
        max_possible_dist = np.sqrt(width**2 + height**2)
        max_current_dist = np.max(self.spatial_dist_continuous)
        fraction_of_max = max_current_dist/max_possible_dist

        ds = max_current_dist/new_max_delay # distance discretized step
        new_dt = ds/config.propagation_vel

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


def linear_activation(neuron_input):
    return neuron_input
