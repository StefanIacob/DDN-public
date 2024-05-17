import gui
import numpy as np
import pickle as pkl

if __name__ == '__main__':

    filename = "2023-05-06energy_efficiency_random_tau_propagation_cost_1e-06.p"
    path = "adaptive_efficient_cma_es_results/" + filename
    with open(path, 'rb') as file:
        data_dict = pkl.load(file)
    print(data_dict.keys())

    gui_save_name = filename[:-2] + "_gui.p"
    gui_path = "visualisations/" + gui_save_name

    net = data_dict['example net']
    params = data_dict['parameters']
    vals = np.mean(data_dict['validation performance'], axis=-1)
    nets_list = []
    print("Building net list...")
    for gen, gen_par in enumerate(params[:5]):
        print('Building gen ' + str(gen))
        best_i = np.argmax(vals[gen])
        best_par = gen_par[best_i, :]
        new_net = net.get_new_network_from_serialized(best_par)
        nets_list.append(new_net)

    # best_param = data_dict['evolutionary strategy'].best.x
    # net = net.get_new_network_from_serialized(best_param)
    #
    print("Building GUI images")
    image_list = []
    for gen, net in enumerate(nets_list):
        print("building image " + str(gen))
        image_list.append(gui.get_network_im(net))

    with open(gui_path, 'wb') as gui_file:
        pkl.dump(image_list, gui_file)
    print("GUI saved as " + gui_path)