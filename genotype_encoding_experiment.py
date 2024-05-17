from populations import tanh_activation
from reservoirpy import datasets
from netGenerator import NetworkGeneratorMLP
import numpy as np
from simulator import NetworkSimulator
from evolution import random_transform_evolution
import os
from datetime import date

if __name__ == '__main__':
    N = 200
    dt = .000005
    buffersize = 20
    n_geno = 500

    gens = 100
    pop_size = 30
    dir = 'random_transform_es_results'
    error_margin = .1

    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1]

    if not os.path.exists(dir):
        os.makedirs(dir)
    tau_range = [12, 22]
    n_range = [10, 10]
    name = str(date.today()) + "_average_fit_" + "_b" + str(buffersize) + "_n" + str(N) + \
           "_multiple_sequences_error_margin_" + str(error_margin)
    print('Running evolution, saving data as ' + str(name))


    netMaker = NetworkGeneratorMLP(N, dt, buffersize, n_geno)

    random_genes = np.random.normal(size=(n_geno,))

    net = netMaker.get_DDN(random_genes)
    sim = NetworkSimulator(net, plasticity=True)
    inp = datasets.mackey_glass(10000)
    sim.visualize(inp[:])

    random_transform_evolution(random_genes, netMaker, 500, 1000, 1000, max_it=gens,
                                          pop_size=pop_size, alphas=alphas, dir=dir, name=name,
                                          n_seq_unsupervised=5,
                                          n_seq_supervised=5,
                                          n_seq_validation=5,
                                          error_margin=error_margin,
                                          tau_range=tau_range,
                                          n_range=n_range,
                                          fitness_function=None)
