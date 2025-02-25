import numpy as np
import cma
from populations import GMMPopulation
from utils import eval_candidate_lag_gridsearch
import pickle
import copy
import os


def cmaes_alg_gma_pop(start_net, train_data, val_data, max_it, pop_size, eval_reps=5, lag_grid=range(0, 15),
                      std=0.3, save_every=1, dir='es_results', name='cma_es_gmm_test', alphas=[10e-14, 10e-13, 10e-12]):

    # order: mix, mu_x, mu_y, var_x, var_y, corr_xy, conn, weight_scaling, bias_scaling, decay

    params = start_net.get_serialized_parameters()
    opts = cma.CMAOptions()
    opts['maxiter'] = max_it
    opts['popsize'] = pop_size
    es = cma.CMAEvolutionStrategy(params, std, opts)

    param_hist = np.zeros((max_it, pop_size, len(params)))
    train_hist = np.zeros((max_it, pop_size, eval_reps, len(lag_grid)))
    val_hist = np.zeros((max_it, pop_size, eval_reps, len(lag_grid)))
    std_hist = np.zeros((max_it,))

    gen = 0

    def save(net):
        data = {
            'validation performance': val_hist,
            'train performance': train_hist,
            'parameters': param_hist,
            'evolutionary strategy': es,
            'cma stds' : std_hist,
            'example net' : net
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
            net_models = {}

            for rep in range(eval_reps):
                # Make sure to resample (i.e. re-generate) a network for every repetition
                new_net = start_net.get_new_network_from_serialized(cand)
                train_scores_lags, val_scores_lags, models_lags = eval_candidate_lag_gridsearch(new_net,
                                                                                                train_data,
                                                                                                val_data,
                                                                                                lag_grid=lag_grid,
                                                                                                alphas=alphas)
                train_hist[gen, c, rep, :] = train_scores_lags
                val_hist[gen, c, rep, :] = val_scores_lags

                # net_models[rep] = {'net': new_net, 'regression models': models_lags}

            # save_net(net_models, gen, c)

        # save every m iterations
        if (gen + 1) % save_every == 0:
            save(new_net)

        val_scores = val_hist[gen, :, :, :]
        best_lags = np.argmin(val_scores, -1)
        best_scores = np.zeros((pop_size, eval_reps))
        for i in range(pop_size):
            for j in range(eval_reps):
                best_scores[i,j] = val_scores[i, j, best_lags[i, j]]

        best_scores = np.mean(best_scores, 1)
        print(best_scores)
        es.tell(candidate_solutions, best_scores)
        print('Gen ', gen)
        gen += 1
    es.result_pretty()
    save()
