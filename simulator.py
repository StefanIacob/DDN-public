from gui import DistDelayGUI
import numpy as np


class NetworkSimulator(object):

    def __init__(self, network, warmup=400):
        self.network = network
        self.warmup_drop = warmup

    def get_network_data(self, input_data):
        net_out = []
        for i, input in enumerate(input_data):
            input = np.ones((len(self.network.neurons_in),)) * input
            self.network.update_step(input)
            network_output_indices = self.network.neurons_out
            if i >= self.warmup_drop:
                output = self.network.A[network_output_indices]
                net_out.append(output)
        net_out = np.stack(net_out, axis=1)
        return net_out.reshape(net_out.shape[:2])

    def visualize(self, input_data):
        gui = DistDelayGUI(self.network)
        for input in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * input
            self.network.update_step(inp)
            gui.update_a()
        gui.close()

    def reset(self):
        self.network.reset_network()
