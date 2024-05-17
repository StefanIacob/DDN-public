from config import propagation_vel
import numpy as np
from populations import scale_up, sigmoid_activation, tanh_activation
from network import DistDelayNetwork


class NetworkGeneratorMLP(object):

    def __init__(self, N, dt, buffersize, n_genotype, n_out=None, act_func=sigmoid_activation):
        if n_out is None:
            n_out = N-1
        diag = (buffersize-1) * dt * propagation_vel
        self.x_range = self.y_range = [0, np.sqrt((diag**2)/2)]

        self.n_type = np.ones((N,))
        self.output_index = np.expand_dims(np.arange(1, n_out + 1),-1)
        self.dt = dt
        self.buffersize = buffersize

        r_weights = [-2, 2]
        r_connectivity = [-1, 1]
        r_bias_scaling = [-2, 2]
        r_loc = self.x_range
        r_decay = [0, 1]
        r_lr = [0, 1]
        r_y0 = [0.2, 2]

        s_weights = (N, N)
        s_connectivity = (N, N)
        s_bias_scaling = (N,)
        s_loc = (N, 2)
        s_decay = (N,)
        s_lr = (N, N)
        s_y0 = (N,)

        self.ranges = [r_weights, r_connectivity, r_bias_scaling, r_loc, r_decay, r_lr, r_y0]
        self.shapes = [s_weights, s_connectivity, s_bias_scaling, s_loc, s_decay, s_lr, s_y0]
        self.hard_lim = [False, False, False, True, True, True, True]

        n_pheno = [np.product(dims) for dims in self.shapes]
        total_pheno = sum(n_pheno)

        n_1 = n_genotype
        n_2 = int(2 * n_genotype)
        n_3 = total_pheno
        self.W12 = np.random.normal(scale=.05, size=(n_1, n_2))
        self.W23 = np.random.normal(scale=.05, size=(n_2, n_3))
        self.act_function = act_func

    def transform(self, genotype):
        centers = [(r[1] + r[0]) / 2 for r in self.ranges]
        scales = [(r[1] - r[0]) / 2 for r in self.ranges]

        phenotype_linear = tanh_activation(tanh_activation(genotype @ self.W12) @ self.W23)
        phenotype_shaped = []
        for i, dims in enumerate(self.shapes):
            n = np.product(dims)
            params = phenotype_linear[:n]
            params = scale_up(params, centers[i], scales[i])
            if self.hard_lim[i]:
                params = np.maximum(params, self.ranges[i][0])
                params = np.minimum(params, self.ranges[i][1])
            phenotype_linear = phenotype_linear[n:]
            params = params.reshape(dims)
            phenotype_shaped.append(params)

        return phenotype_shaped

    def get_DDN(self, genotype):
        parameters = self.transform(genotype)

        W = parameters[0]
        connectivity = parameters[1] > 0
        bias = parameters[2]
        coordinates = parameters[3]
        n_decay = parameters[4]
        lr = parameters[5]
        y0 = parameters[6]

        coordinates[0, 0] = 0
        coordinates[0, 0] = 0

        W = W * connectivity

        net = DistDelayNetwork(weights=W, bias=bias, n_type=self.n_type, coordinates=coordinates, decay=n_decay, input_n=[0],
                         output_n=self.output_index, activation_func=self.act_function, dt=self.dt, buffersize=self.buffersize,
                         theta_window=self.buffersize, x_range=self.x_range, y_range=self.y_range, lr=lr, theta_y0=y0)

        return net
