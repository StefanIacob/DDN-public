import numpy as np
import cma
from utils import eval_candidate_signal_gen_multiple_random_sequences_adaptive
import pickle
import os

def cmaes_alg_gma_pop_signal_gen_adaptive(start_net, n_unsupervised,
                                          n_supervised,
                                          n_validation, max_it, pop_size,
                                          eval_reps=5, std=0.3,
                                          save_every=1,
                                          dir='es_results', name='cma_es_gmm_test', alphas=[10e-14, 10e-13, 10e-12],
                                          n_seq_unsupervised=5, n_seq_supervised=5, n_seq_validation=5, error_margin=.1,
                                          tau_range=[12, 22], n_range=[5,15], fitness_function=None):
    params = start_net.get_serialized_parameters()
    opts = cma.CMAOptions()
    opts['maxiter'] = max_it
    opts['popsize'] = pop_size
    es = cma.CMAEvolutionStrategy(params, std, opts)

    param_hist = np.zeros((max_it, pop_size, len(params)))
    val_hist = np.zeros((max_it, pop_size, eval_reps))
    std_hist = np.zeros((max_it,))

    gen = 0
    x0_range = [.5, 1.2]

    def save(net):
        data = {
            'validation performance': val_hist,
            'parameters': param_hist,
            'evolutionary strategy': es,
            'cma stds': std_hist,
            'error margin': error_margin,
            'number of sequences': {
                    'unsupervised': n_seq_unsupervised,
                    'supervised': n_seq_supervised,
                    'validation': n_seq_validation,
            },
            'number of samples': {
                'unsupervised': n_unsupervised,
                'supervised': n_supervised,
                'validation': n_validation
            },
            'alpha grid': alphas,
            'tau range': tau_range,
            'start value range': x0_range,
            'example net': net
        }
        file = open(dir + '/' + name + '.p', "wb")
        pickle.dump(data, file)
        file.close()

    def save_net(data, gen, c):
        net_dir = dir + '/' + name + '_nets/gen_' + str(gen)
        if not os.path.exists(net_dir):
            os.makedirs(net_dir)
        nets_file = dir + '/' + name + '_nets/gen_' + str(gen) + '/' + str(c) + '.p'
        nets_file = open(nets_file, "wb")
        pickle.dump(data, nets_file)
        nets_file.close()

    random_gen = np.random.default_rng(seed=42)
    while not es.stop():
        candidate_solutions = es.ask()
        for c, cand in enumerate(candidate_solutions):
            # train_score_cand = np.zeros((nr_of_evals, len(lag_grid)))
            # val_score_cand = np.zeros((nr_of_evals, len(lag_grid)))

            param_hist[gen, c, :] = cand
            std_hist[gen] = es.sigma
            # net_models = {}

            for rep in range(eval_reps):
                # Make sure to resample (i.e. re-generate) a network for every repetition
                new_net = start_net.get_new_network_from_serialized(cand)
                validation_horizon, model, new_net = eval_candidate_signal_gen_multiple_random_sequences_adaptive(new_net,
                                                                                                         n_seq_unsupervised,
                                                                                                         n_seq_supervised,
                                                                                                         n_seq_validation,
                                                                                                         n_unsupervised,
                                                                                                         n_supervised,
                                                                                                         n_validation,
                                                                                                         alphas=alphas,
                                                                                                         error_margin=error_margin,
                                                                                                         tau_range=tau_range,
                                                                                                         n_range=n_range,
                                                                                                         x0_range=x0_range)
                val_hist[gen, c, rep] = validation_horizon

        # save every m iterations
        if (gen + 1) % save_every == 0:
            save(new_net)

        if fitness_function is None:
            fitness = np.mean(val_hist[gen, :, :], axis=-1)
        else:
            fitness = fitness_function(val_hist[gen, :, :])

        print(fitness)

        es.tell(candidate_solutions, 1 / fitness)
        print('Gen ', gen)
        gen += 1
    es.result_pretty()
