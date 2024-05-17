from gui import DistDelayGUI, DistDelayGUI_arm
import numpy as np
import matplotlib.pyplot as plt
from populations import GMMPopulationAdaptive
import network

class NetworkArmSimulator(object):

    def __init__(self, network, arm, plasticity=True):
        self.network = network
        self.arm = arm
        self.plasticity = False
        self.gain = 20
        if type(self.network) is GMMPopulationAdaptive:
            self.plasticity = plasticity

    def scale_neuron2torque(self, activation, gain=20):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        center_neuron = (range_neuron[0] + range_neuron[1]) / 2
        scale_neuron = (range_neuron[1] - range_neuron[0]) / 2
        torque = (activation - center_neuron) * scale_neuron * gain
        return torque

    def scale_q2neuron(self, q):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        range_q = (0.1, 3.0)
        center_q = (range_q[0] + range_q[1])/2
        scale_q = (range_q[1] - range_q[0])/2
        center_neuron = (range_neuron[0] + range_neuron[1])/2
        scale_neuron = (range_neuron[1] - range_neuron[0])/2
        q = q - center_q
        q = (q / scale_q) * scale_neuron
        q += center_neuron
        return np.clip(q, range_neuron[0], range_neuron[1])

    def scale_dq2neuron(self, dq):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        range_dq = (-6, 6)
        center_dq = (range_dq[0] + range_dq[1])/2
        scale_dq = (range_dq[1] - range_dq[0])/2
        center_neuron = (range_neuron[0] + range_neuron[1])/2
        scale_neuron = (range_neuron[1] - range_neuron[0])/2
        dq = dq - center_dq
        dq = (dq / scale_dq) * scale_neuron
        dq += center_neuron
        return np.clip(dq, range_neuron[0], range_neuron[1])

    def sim_step(self, target):
        q = np.array(self.arm.q)
        dq = np.array(self.arm.dq)
        q = self.scale_q2neuron(q)
        dq = self.scale_dq2neuron(dq)
        net_in = np.concatenate([q, dq, target])
        self.network.update_step(net_in)
        net_out = self.network.A[self.network.neurons_out, 0]
        net_out = self.scale_neuron2torque(net_out, self.gain)
        self.arm.arm_func(net_out)

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

    def visualize_arm(self, targets):
        gui = DistDelayGUI_arm(self.network, self.arm)

        for t in targets:
            self.sim_step(t)
            gui.update_a(t)

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
            # if j == 100:
            #     j = 0
            #     if self.network.buffersize > 1:
            #         Wds = np.array(self.network.W_masked_list)
            #         newW = np.sum(Wds, axis=0)
            #     else:
            #         newW = self.network.W
            #     nonzeroNewW = list(np.reshape(newW, newW.shape[0] * newW.shape[1]))
            #     nonzeroNewW = [i for i in nonzeroNewW if i != 0]
            #     print('min (nonzero) ', np.min(nonzeroNewW))
            #     print('max (nonzero) ', np.max(nonzeroNewW))
            #     print('average (nonzero) ', np.average(nonzeroNewW))
            #     print('std (nonzero) ', np.std(nonzeroNewW))
            #     print('N nonzero weights ', len(nonzeroNewW))
            #     print('norm', np.linalg.norm(newW))
            #     # plt.hist(nonzeroNewW, bins=20)
            #     # plt.pause(0.01)
        gui.close()

    def reset(self):
        self.network.reset_activity()

    def full_reset(self):
        self.network.reset_network()


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
            # if j == 100:
            #     j = 0
            #     if self.network.buffersize > 1:
            #         Wds = np.array(self.network.W_masked_list)
            #         newW = np.sum(Wds, axis=0)
            #     else:
            #         newW = self.network.W
            #     nonzeroNewW = list(np.reshape(newW, newW.shape[0] * newW.shape[1]))
            #     nonzeroNewW = [i for i in nonzeroNewW if i != 0]
            #     print('min (nonzero) ', np.min(nonzeroNewW))
            #     print('max (nonzero) ', np.max(nonzeroNewW))
            #     print('average (nonzero) ', np.average(nonzeroNewW))
            #     print('std (nonzero) ', np.std(nonzeroNewW))
            #     print('N nonzero weights ', len(nonzeroNewW))
            #     print('norm', np.linalg.norm(newW))
            #     # plt.hist(nonzeroNewW, bins=20)
            #     # plt.pause(0.01)
        gui.close()

    def reset(self):
        self.network.reset_activity()

    def full_reset(self):
        self.network.reset_network()
