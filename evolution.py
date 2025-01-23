import numpy as np
import cma
from utils import eval_candidate_lag_gridsearch_NARMA, eval_candidate_signal_gen, eval_candidate_signal_gen_adaptive, \
    eval_candidate_signal_gen_horizon, \
    eval_candidate_signal_gen_multiple_random_sequences_adaptive_budget, eval_candidate_custom_data_signal_gen
import pickle
import os
from reservoirpy import datasets

def cmaes_alg_gma_pop_timeseries_prediction(start_net, train_data, val_data, max_it, pop_size, eval_reps=5,
                                            lag_grid=range(0, 15), std=0.3, save_every=1, dir='es_results',
                                            name='cma_es_gmm_test', alphas=[10e-14, 10e-13, 10e-12]):
    # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, bias_scaling, decay

    params = start_net.get_serialized_parameters()
    opts = cma.CMAOptions()
    opts['maxiter'] = max_it
    opts['popsize'] = pop_size
    es = cma.CMAEvolutionStrategy(params, std, opts)

    param_hist = np.zeros((max_it, pop_size, len(params)))
    val_hist = np.zeros((max_it, pop_size, eval_reps, len(lag_grid)))
    std_hist = np.zeros((max_it,))

    gen = 0

    def save(net):
        data = {
            'validation performance': val_hist,
            'parameters': param_hist,
            'evolutionary strategy': es,
            'cma stds': std_hist,
            'example net': net,
            'train data': train_data,
            'validation data': val_data,
            'alpha grid': alphas,
            'start net': start_net
        }
        file = open(dir + '/' + name + '.p', "wb")
        pickle.dump(data, file)
        file.close()

    while not es.stop():
        candidate_solutions = es.ask()
        for c, cand in enumerate(candidate_solutions):
            param_hist[gen, c, :] = cand
            std_hist[gen] = es.sigma

            for rep in range(eval_reps):
                # Make sure to resample (i.e. re-generate) a network for every repetition
                new_net = start_net.get_new_network_from_serialized(cand)
                _, val_scores_lags, _ = eval_candidate_lag_gridsearch_NARMA(new_net,
                                                                              train_data,
                                                                              val_data,
                                                                              lag_grid=lag_grid,
                                                                              alphas=alphas)
                val_hist[gen, c, rep, :] = val_scores_lags

        # save every m iterations
        if (gen + 1) % save_every == 0:
            save(new_net)

        val_scores = val_hist[gen, :, :, :]
        best_lags = np.argmin(val_scores, -1)
        best_scores = np.zeros((pop_size, eval_reps))
        for i in range(pop_size):
            for j in range(eval_reps):
                best_scores[i, j] = val_scores[i, j, best_lags[i, j]]

        best_scores = np.mean(best_scores, 1)
        print(best_scores)
        es.tell(candidate_solutions, best_scores)
        print('Gen ', gen)
        gen += 1
    es.result_pretty()

def cmaes_alg_gma_pop_signal_gen(start_net, train_data, val_data, max_it, pop_size, eval_reps=5, std=0.3, save_every=1,
                                 dir='es_results', name='cma_es_gmm_test', alphas=[10e-14, 10e-13, 10e-12]):
    # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, bias_scaling, decay

    params = start_net.get_serialized_parameters()
    opts = cma.CMAOptions()
    opts['maxiter'] = max_it
    opts['popsize'] = pop_size
    # opts['ftarget'] = float('inf')
    es = cma.CMAEvolutionStrategy(params, std, opts)

    param_hist = np.zeros((max_it, pop_size, len(params)))
    val_hist = np.zeros((max_it, pop_size, eval_reps))
    std_hist = np.zeros((max_it,))

    gen = 0

    def save(net):
        data = {
            'validation performance': val_hist,
            'parameters': param_hist,
            'evolutionary strategy': es,
            'cma stds': std_hist,
            'example net': net,
            'alpha grid': alphas,
            'data': {
                'train': train_data,
                'validation': val_data,
            }
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
                validation_horizon, model = eval_candidate_signal_gen(new_net,
                                                                      train_data,
                                                                      val_data,
                                                                      alphas=alphas)
                val_hist[gen, c, rep] = validation_horizon
                # net_models[rep] = {'net': new_net, 'regression models': models_lags}

            # save_net(net_models, gen, c)
        # save every m iterations
        if (gen + 1) % save_every == 0:
            save(new_net)

        fitness = np.mean(val_hist[gen, :, :], axis=-1)
        print(fitness)
        es.tell(candidate_solutions, 1 / fitness)
        print('Gen ', gen)
        gen += 1
    es.result_pretty()

def continue_cmaes_adaptive(dir, name,
                            eval_reps=5, save_every=1, fitness_function=None):

    cmaes_data_filename = dir + '/' + name + '.p'
    data_file = open(cmaes_data_filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    es = data['evolutionary strategy']
    param_hist = data['parameters']
    val_hist = data['validation performance']
    start_net = data['example net']
    std_hist = data['cma stds']
    n_seq_unsupervised = data['number of sequences']['unsupervised']
    n_seq_supervised = data['number of sequences']['supervised']
    n_seq_validation = data['number of sequences']['validation']
    n_unsupervised = data['number of samples']['unsupervised']
    n_supervised = data['number of samples']['supervised']
    n_validation = data['number of samples']['validation']
    alphas = data['alpha grid']
    error_margin = data["error margin"]
    tau_range = data["tau range"]
    x0_range = data["start value range"]
    n_range = [10, 10]

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

    random_gen = np.random.default_rng(seed=42)
    gen = es.countiter + 1
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
                # unsupervised_data = datasets.mackey_glass(n_unsupervised, x0=np.random.uniform(0.5, 1.5), seed=random_gen)
                # supervised_data = datasets.mackey_glass(n_supervised, x0=np.random.uniform(0.5, 1.5), seed=random_gen)
                # val_data = datasets.mackey_glass(n_validation, x0=np.random.uniform(0.5, 1.5), seed=random_gen)
                # validation_horizon, model = eval_candidate_signal_gen_adaptive(new_net,
                #                                                                unsupervised_data,
                #                                                                supervised_data,
                #                                                                val_data,
                #                                                                alphas=alphas)
                validation_horizon, model, new_net = eval_candidate_signal_gen_horizon(
                    new_net,
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
                # net_models[rep] = {'net': new_net, 'regression models': models_lags}

            # save_net(net_models, gen, c)
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


def cmaes_alg_gma_pop_signal_gen_adaptive(start_net, n_unsupervised,
                                          n_supervised,
                                          n_validation, max_it, pop_size,
                                          eval_reps=5, std=0.3,
                                          save_every=1,
                                          dir='es_results', name='cma_es_gmm_test', alphas=[10e-14, 10e-13, 10e-12],
                                          n_seq_unsupervised=5, n_seq_supervised=5, n_seq_validation=5, error_margin=.1,
                                          tau_range=[12, 22], n_range=[5,15], fitness_function=None, activation_cost=.005,
                                          synapse_cost=.001,
                                          propagation_cost=.005, aggregate=np.mean):
    params = start_net.get_serialized_parameters()
    opts = cma.CMAOptions()
    opts['maxiter'] = max_it
    opts['popsize'] = pop_size
    es = cma.CMAEvolutionStrategy(params, std, opts)

    param_hist = np.zeros((max_it, pop_size, len(params)))
    val_hist = np.zeros((max_it, pop_size, eval_reps))
    energy_hist = np.zeros((max_it, pop_size, eval_reps))
    std_hist = np.zeros((max_it,))

    gen = 0
    x0_range = [.5, 1.2]

    def save(net):
        data = {
            'validation performance': val_hist,
            'energy consumption': energy_hist,
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
                validation_horizon, model, energy = eval_candidate_signal_gen_horizon(new_net, n_seq_unsupervised,
                                                                                      n_seq_supervised, n_seq_validation,
                                                                                      n_unsupervised, n_supervised,
                                                                                      n_validation, alphas=alphas,
                                                                                      tau_range=tau_range, n_range=n_range,
                                                                                      x0_range=x0_range)
                # validation_horizon, model, energy = eval_candidate_signal_gen_multiple_random_sequences_adaptive_budget(new_net,
                #                                                                                          n_seq_unsupervised,
                #                                                                                          n_seq_supervised,
                #                                                                                          n_seq_validation,
                #                                                                                          n_unsupervised,
                #                                                                                          n_supervised,
                #                                                                                          n_validation,
                #                                                                                          alphas=alphas,
                #                                                                                          tau_range=tau_range,
                #                                                                                          n_range=n_range,
                #                                                                                          x0_range=x0_range,
                #                                                                                          activation_cost=activation_cost,
                #                                                                                          synapse_cost=synapse_cost,
                #                                                                                          propagation_cost=propagation_cost
                #                                                                                                     )
                val_hist[gen, c, rep] = validation_horizon
                energy_hist[gen, c, rep] = energy
                # net_models[rep] = {'net': new_net, 'regression models': models_lags}

            # save_net(net_models, gen, c)
        # save every m iterations
        if (gen + 1) % save_every == 0:
            save(new_net)

        if fitness_function is None:
            fitness = aggregate(val_hist[gen, :, :], axis=-1)
        else:
            fitness = fitness_function(val_hist[gen, :, :], energy_hist[gen, :, :])

        print(fitness)

        es.tell(candidate_solutions, 1 / fitness)
        print('Gen ', gen)
        gen += 1
    es.result_pretty()

def cmaes_signal_gen_adaptive_custom_datasets(start_net, unsupervised, supervised, validation, pop_size, max_it=99, eval_reps=5,
                                              std=0.3, save_every=1, dir='es_results', name='cma_es_gmm_test',
                                              alphas=[10e-14, 10e-13, 10e-12], error_margin=.1, fitness_function=None,
                                              warmup_overlap=False):
    params = start_net.get_serialized_parameters()
    opts = cma.CMAOptions()
    opts['maxiter'] = max_it
    opts['popsize'] = pop_size
    es = cma.CMAEvolutionStrategy(params, std, opts)

    param_hist = np.zeros((max_it, pop_size, len(params)))
    val_hist = np.zeros((max_it, pop_size, eval_reps))
    std_hist = np.zeros((max_it,))
    shuffle = True
    if warmup_overlap:
        shuffle = False

    gen = 0

    def save(net):
        data = {
            'validation performance': val_hist,
            'parameters': param_hist,
            'evolutionary strategy': es,
            'cma stds': std_hist,
            'error margin': error_margin,
            'example net': net
        }
        file = open(dir + '/' + name + '.p', "wb")
        pickle.dump(data, file)
        file.close()

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
                validation_horizon, model = eval_candidate_custom_data_signal_gen(new_net, unsupervised, supervised,
                                                                                  validation, error_margin=error_margin,
                                                                                  alphas=alphas, warmup_overlap=warmup_overlap,
                                                                                  shuffle=shuffle)
                val_hist[gen, c, rep] = validation_horizon
                # net_models[rep] = {'net': new_net, 'regression models': models_lags}

            # save_net(net_models, gen, c)
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

def random_transform_evolution(params, net_maker,
                                          n_unsupervised,
                                          n_supervised,
                                          n_validation, max_it, pop_size,
                                          eval_reps=5, std=0.3,
                                          save_every=1,
                                          dir='es_results', name='cma_es_gmm_test', alphas=[10e-14, 10e-13, 10e-12],
                                          n_seq_unsupervised=5, n_seq_supervised=5, n_seq_validation=5, error_margin=.1,
                                          tau_range=[12, 22], n_range=[5,15], fitness_function=None):
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
            'net maker': net_maker
        }
        file = open(dir + '/' + name + '.p', "wb")
        pickle.dump(data, file)
        file.close()


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
                new_net = net_maker.get_DDN(cand)
                # unsupervised_data = datasets.mackey_glass(n_unsupervised, x0=np.random.uniform(0.5, 1.5), seed=random_gen)
                # supervised_data = datasets.mackey_glass(n_supervised, x0=np.random.uniform(0.5, 1.5), seed=random_gen)
                # val_data = datasets.mackey_glass(n_validation, x0=np.random.uniform(0.5, 1.5), seed=random_gen)
                # validation_horizon, model = eval_candidate_signal_gen_adaptive(new_net,
                #                                                                unsupervised_data,
                #                                                                supervised_data,
                #                                                                val_data,
                #                                                                alphas=alphas)
                validation_horizon, model, new_net = eval_candidate_signal_gen_horizon(new_net,
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
                # net_models[rep] = {'net': new_net, 'regression models': models_lags}

            # save_net(net_models, gen, c)
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
