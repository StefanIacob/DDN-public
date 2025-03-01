from gui import DistDelayGUI
import numpy as np
from populations import GMMPopulationAdaptive


class NetworkSimulator(object):

    def __init__(self, network, plasticity=True):
        self.network = network
        self.plasticity = False
        if type(self.network) is GMMPopulationAdaptive:
            self.plasticity = plasticity

    def warmup(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)

    def unsupervised(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step_adaptive(inp)

    def get_network_data(self, input_data):
        net_out = []
        network_output_indices = self.network.neurons_out
        for i, inp in enumerate(input_data):
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)
            output = self.network.A[network_output_indices, 0]
            net_out.append(output)
        net_out = np.stack(net_out, axis=1)
        return net_out.reshape(net_out.shape[:2])

    def visualize_feedback(self, start_value, readout_model, labels):
        gui = DistDelayGUI(self.network)
        inp = start_value
        network_output_indices = self.network.neurons_out

        for l in labels:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)
            output = self.network.A[network_output_indices, 0].T
            inp = readout_model.predict(output)[0,0]
            abs_error = np.abs(l - inp)
            print('Absolute error:', abs_error)
            gui.update_a()

    def visualize(self, input_data):
        gui = DistDelayGUI(self.network)
        j = 0

        for i, input in enumerate(input_data):
            j += 1
            inp = np.ones((len(self.network.neurons_in),)) * input
            if self.plasticity:
                self.network.update_step_adaptive(inp)
            else:
                self.network.update_step(inp)

            gui.update_a()

        gui.close()

    def reset(self):
        self.network.reset_activity()

    def full_reset(self):
        self.network.reset_network()

