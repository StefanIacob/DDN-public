import numpy as np
from simulator import NetworkSimulator
from scipy.stats import norm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pickle as pkl
from reservoirpy import datasets
from populations import GMMPopulationAdaptive
import random

def read_config(config_parser):

    config_dict = {}
    for section in config_parser.sections():
        section_dict = {}
        for key, val in config_parser.items(section):
            if not (section == 'save file' or key == 'activation' or key == 'growing' or section == 'fixed genome'):
                if "," in val:
                    val = list(map(float, val.split(',')))
                else:
                    val = float(val)
            elif section == 'fixed genome' or key == 'growing':
                val = val == "True"
            section_dict[key] = val
        config_dict[section] = section_dict

    return config_dict

def mse(target_signal, input_signal):
    """
    rmse(input_signal, target_signal)-> error
    MSE calculation.
    Calculates the mean square error (MSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    # target_signal = target_signal.view(target_signal.numel())
    # input_signal = input_signal.view(input_signal.numel())

    error = (target_signal - input_signal) ** 2
    return error.mean()


def nmse(target_signal, input_signal):
    """
    nmse(input_signal, target_signal)-> error
    NMSE calculation.
    Calculates the normalized mean square error (NMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    # check_signal_dimensions(input_signal, target_signal)

    if len(target_signal) == 1:
        raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')

    # Use normalization with N-1, as in matlab
    var = np.std(target_signal) ** 2

    return mse(target_signal, input_signal) / var


def nrmse(input_signal, target_signal):
    """
    nrmse(input_signal, target_signal)-> error
    NRMSE calculation.
    Calculates the normalized root mean square error (NRMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    # check_signal_dimensions(input_signal, target_signal)

    if len(target_signal) == 1:
        raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')

    return np.sqrt(nmse(target_signal, input_signal))


def state_entropy(A):
    N = len(A)
    std = np.std(A)
    state_kernel = []
    for i in range(N):
        for j in range(N):
            kern = norm.pdf(A[i] - A[i], 0, 0.3 * std)
            state_kernel.append(kern)
    H2 = -1 * np.log((1 / N ** 2) * np.sum(state_kernel))
    return H2


def spectral_radius_norm(W, wanted_sr):
    v = np.linalg.eigvals(W)
    sr = np.max(np.absolute(v))
    W_scaled = (W * wanted_sr) / sr
    return W_scaled

def inputs2NARMA(inputs, system_order=10, coef=[.3, .05, 1.5, .1]):
    length = len(inputs)
    outputs = np.zeros((length, 1))
    for k in range(system_order - 1, length - 1):
        outputs[k + 1] = coef[0] * outputs[k] + coef[1] * \
                         outputs[k] * np.sum(outputs[k - (system_order - 1):k + 1]) + \
                         coef[2] * inputs[k - (system_order - 1)] * inputs[k] + coef[3]
    return outputs

def createNARMA(length=10000, system_order=10, coef=[.3, .05, 1.5, .1]):
    inputs = np.random.rand(length, 1) * .5
    inputs.shape = (-1, 1)
    outputs = np.zeros((length, 1))
    for k in range(system_order - 1, length - 1):
        outputs[k + 1] = coef[0] * outputs[k] + coef[1] * \
                         outputs[k] * np.sum(outputs[k - (system_order - 1):k + 1]) + \
                         coef[2] * inputs[k - (system_order - 1)] * inputs[k] + coef[3]
    return inputs, outputs


def createNARMA10(length=10000):
    return createNARMA(length=length, system_order=10, coef=[.3, .05, 1.5, .1])


def createNARMA30(length=10000):
    return createNARMA(length=length, system_order=30, coef=[.2, .04, 1.5, .001])

def eval_candidate_lag_gridsearch_NARMA(network, train_data, val_data, warmup=400,
                                        lag_grid=range(0, 15), alphas=[10e-14, 10e-13, 10e-12]):
    assert np.all(np.array(lag_grid) >= 0), 'No negative lag allowed'

    train_input = train_data[0, warmup:]
    train_input_warmup = train_data[0, :warmup]
    train_labels = train_data[1, warmup:]

    val_input = val_data[0, warmup:]
    val_input_warmup = val_data[0, :warmup]
    val_labels = val_data[1, warmup:]
    sim = NetworkSimulator(network)

    # generate net activity
    # run warmup
    sim.warmup(train_input_warmup)
    train_net_act = sim.get_network_data(train_input).T
    sim.reset()
    # run warmup
    sim.warmup(val_input_warmup)
    val_net_act = sim.get_network_data(val_input).T

    val_performance_per_lag = []
    train_performance_per_lag = []
    model_per_lag = {}

    for lag in lag_grid:
        model = RidgeCV(alphas=alphas, cv=5)

        # clip labels to fit lag parameter
        train_labels_clipped = train_labels
        val_labels_clipped = val_labels
        if lag > 0:
            train_labels_clipped = train_labels[:-lag]
            val_labels_clipped = val_labels[:-lag]

        model.fit(train_net_act[lag:], train_labels_clipped)

        train_predictions = model.predict(train_net_act)
        train_performance = nrmse(train_predictions[lag:], train_labels_clipped)

        # test readout
        val_predictions = model.predict(val_net_act)
        val_performance = nrmse(val_predictions[lag:], val_labels_clipped)

        val_performance_per_lag.append(val_performance)
        train_performance_per_lag.append(train_performance)
        model_per_lag[lag] = model



    return train_performance_per_lag, val_performance_per_lag, model_per_lag

def eval_candidate_lag_gridsearch_NARMA_multitask(network, input_train, input_val, labels_train, labels_val, warmup=400,
                                        lag_grid=range(0, 15), alphas=[10e-14, 10e-13, 10e-12]):
    assert np.all(np.array(lag_grid) >= 0), 'No negative lag allowed'
    assert type(input_train) == type(input_val) == np.ndarray
    assert type(labels_train) == type(labels_val) == list
    n_tasks = len(labels_train)
    train_input = input_train[warmup:]
    train_input_warmup = input_train[:warmup]

    val_input = input_val[warmup:]
    val_input_warmup = input_val[:warmup]

    sim = NetworkSimulator(network)

    # generate net activity
    # run warmup
    sim.warmup(train_input_warmup)
    train_net_act = sim.get_network_data(train_input).T
    sim.reset()
    # run warmup
    sim.warmup(val_input_warmup)
    val_net_act = sim.get_network_data(val_input).T

    val_performance_per_task = []
    train_performance_per_task = []
    model_per_task = {}

    for task_i, train_labels in enumerate(labels_train):
        train_labels = train_labels[warmup:]
        val_labels = labels_val[task_i][warmup:]
        val_performance_per_lag = []
        train_performance_per_lag = []
        model_per_lag = {}

        for lag in lag_grid:
            model = RidgeCV(alphas=alphas, cv=5)

            # clip labels to fit lag parameter
            train_labels_clipped = train_labels
            val_labels_clipped = val_labels
            if lag > 0:
                train_labels_clipped = train_labels[:-lag]
                val_labels_clipped = val_labels[:-lag]

            model.fit(train_net_act[lag:], train_labels_clipped)

            train_predictions = model.predict(train_net_act)
            train_performance = nrmse(train_predictions[lag:], train_labels_clipped)

            # test readout
            val_predictions = model.predict(val_net_act)
            val_performance = nrmse(val_predictions[lag:], val_labels_clipped)

            val_performance_per_lag.append(val_performance)
            train_performance_per_lag.append(train_performance)
            model_per_lag[lag] = model
        val_performance_per_task.append(val_performance_per_lag)
        train_performance_per_task.append(train_performance_per_lag)
        model_per_task[task_i] = model_per_lag

    return train_performance_per_task, val_performance_per_task, model_per_task



def eval_candidate_signal_gen(network, train_data, val_data, error_margin=.1, max_it_val=500, warmup=400,
                              alphas=[10e-14, 10e-13, 10e-12]):
    # Training: train with one step ahead prediction
    sim = NetworkSimulator(network)
    warmup_input_train = train_data[:warmup]
    input_train = train_data[warmup:-1]
    labels_train = train_data[warmup + 1:]
    sim.warmup(warmup_input_train)
    net_act_train = sim.get_network_data(input_train).T
    sim.reset()
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(net_act_train, labels_train)

    # Validation: fitness measure is amount of prediction steps within error margin
    warmup_input_val = val_data[:warmup]
    start_input_val = val_data[warmup]
    labels_val = val_data[warmup + 1:]
    sim.warmup(warmup_input_val)

    feedback_in = start_input_val
    error = 0
    i = 0
    label_variance = np.var(labels_val)
    while error < error_margin and i <= max_it_val:
        feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
        network.update_step(feedback_in)
        output = network.A[network.neurons_out, 0].T
        feedback_in = model.predict(output)[0][0]
        error = single_sample_NRSE(feedback_in, labels_val[i, 0],
                                   label_variance)  # np.abs(labels_val[i, 0] - feedback_in)
        i += 1

    return i, model


def eval_candidate_signal_gen_adaptive(network, unsupervised_data, train_data, val_data, error_margin=.1,
                                       max_it_val=500,
                                       warmup=400, alphas=[10e-14, 10e-13, 10e-12]):
    sim = NetworkSimulator(network)

    # Unsupervised: run with synaptic plasticity
    warmup_unsupervised = unsupervised_data[:warmup]
    input_unsupervised = unsupervised_data[warmup:]
    sim.warmup(warmup_unsupervised)
    sim.unsupervised(input_unsupervised)

    # Training: train with one step ahead prediction
    warmup_train = train_data[:warmup]
    input_train = train_data[warmup : -1]
    labels_train = train_data[warmup + 1:]
    sim.warmup(warmup_train)
    net_act_train = sim.get_network_data(input_train).T
    sim.reset()
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(net_act_train, labels_train)

    # Validation: fitness measure is amount of prediction steps within error margin
    warmup_input_val = val_data[:warmup]
    start_input_val = val_data[warmup]
    labels_val = val_data[warmup + 1:]
    sim.warmup(warmup_input_val)

    feedback_in = start_input_val
    error = 0
    i = 0
    label_variance = np.var(labels_val)
    while error < error_margin and i <= max_it_val:
        feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
        network.update_step(feedback_in)
        output = network.A[network.neurons_out, 0].T
        feedback_in = model.predict(output)[0][0]
        error = single_sample_NRSE(feedback_in, labels_val[i, 0],
                                   label_variance)  # np.abs(labels_val[i, 0] - feedback_in)
        i += 1

    return i, model


def create_folds(X, y, n_folds, groups):
    folds = []
    cv_object = KFold(n_splits = n_folds)
    for (train_indices, val_indices) in cv_object.split(X, y, groups=groups):
        folds.append((train_indices,val_indices))
    return folds


def eval_candidate_signal_gen_NRMSE(network, n_sequences_unsupervised,
                                      n_sequences_supervised,
                                      n_sequences_validation,
                                      n_unsupervised,
                                      n_supervised,
                                      n_validation,
                                      error_margin=.1, max_it_val=500, warmup=400,
                                      alphas=[10e-14, 10e-13, 10e-12],
                                      seed=42,
                                      tau_range=[12, 22],
                                      n_range=[5, 15],
                                      x0_range=[.5, 1.2]):
    sim = NetworkSimulator(network)
    random_gen = np.random.default_rng(seed=seed)
    random_tau = np.random.uniform(tau_range[0], tau_range[1])
    random_exp = np.random.uniform(n_range[0], n_range[1])

    if n_sequences_unsupervised > 0 and type(network) is GMMPopulationAdaptive:
        # Unsupervised: run with synaptic plasticity on different sequences
        for i in range(n_sequences_unsupervised):
            data = datasets.mackey_glass(n_unsupervised + warmup, tau=random_tau, n=random_exp,
                                         x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
            sim.warmup(data[:warmup])
            sim.unsupervised(data[warmup:])
            sim.reset()

    # Supervised: one step ahead teacher forcing with fixed weights
    net_act_train_across_sequences = []
    labels_train_across_sequences = []
    groups = []
    for i in range(n_sequences_supervised):
        data = datasets.mackey_glass(n_supervised + warmup, tau=random_tau, n=random_exp,
                                     x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
        input = data[warmup:-1]
        labels = data[warmup + 1:]
        sim.warmup(data[:warmup])
        net_act = sim.get_network_data(input)
        sim.reset()
        net_act_train_across_sequences.append(net_act)
        labels_train_across_sequences.append(labels)
        groups.append(np.ones_like(input) * i)

    net_act_train_across_sequences = np.concatenate(net_act_train_across_sequences, axis=1)
    labels_train_across_sequences = np.concatenate(labels_train_across_sequences, axis=0)
    groups = np.concatenate(groups, axis=0)

    folds = create_folds(net_act_train_across_sequences.T, labels_train_across_sequences,
                 10, groups)
    model = RidgeCV(alphas=alphas, cv=folds)
    model.fit(net_act_train_across_sequences.T, labels_train_across_sequences)

    prediction_steps_across_sequences = []
    energy_use_across_sequences = 0

    # validation
    for i in range(n_sequences_validation):
        data = datasets.mackey_glass(n_validation + warmup, tau=random_tau, n=random_exp,
                                     x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
        start_input_val = data[warmup]
        labels_val = data[warmup + 1:]
        sim.warmup(data[:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val
        label_variance = np.var(labels_val)

        while error <= error_margin and j <= max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            power = network.get_current_power_bio()
            energy_use_across_sequences += power * network.dt
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            j += 1

        prediction_steps_across_sequences.append(j)

    return np.mean(prediction_steps_across_sequences), model, energy_use_across_sequences


def eval_candidate_signal_gen_horizon(network, n_sequences_unsupervised,
                                      n_sequences_supervised,
                                      n_sequences_validation,
                                      n_unsupervised,
                                      n_supervised,
                                      n_validation,
                                      error_margin=.1, max_it_val=500, warmup=400,
                                      alphas=[10e-14, 10e-13, 10e-12],
                                      seed=42,
                                      tau_range=[12, 22],
                                      n_range=[5, 15],
                                      x0_range=[.5, 1.2],
                                      aggregate=np.mean):
    sim = NetworkSimulator(network)
    random_gen = np.random.default_rng(seed=seed)
    random_tau = np.random.uniform(tau_range[0], tau_range[1])
    random_exp = np.random.uniform(n_range[0], n_range[1])

    if n_sequences_unsupervised > 0: # and type(network) is GMMPopulationAdaptive:
        # Unsupervised: run with synaptic plasticity on different sequences
        for i in range(n_sequences_unsupervised):
            data = datasets.mackey_glass(n_unsupervised + warmup, tau=random_tau, n=random_exp,
                                         x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
            sim.warmup(data[:warmup])
            sim.unsupervised(data[warmup:])
            sim.reset()

    # Supervised: one step ahead teacher forcing with fixed weights
    net_act_train_across_sequences = []
    labels_train_across_sequences = []
    groups = []
    for i in range(n_sequences_supervised):
        data = datasets.mackey_glass(n_supervised + warmup, tau=random_tau, n=random_exp,
                                     x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
        input = data[warmup:-1]
        labels = data[warmup + 1:]
        sim.warmup(data[:warmup])
        net_act = sim.get_network_data(input)
        sim.reset()
        net_act_train_across_sequences.append(net_act)
        labels_train_across_sequences.append(labels)
        groups.append(np.ones_like(input) * i)

    net_act_train_across_sequences = np.concatenate(net_act_train_across_sequences, axis=1)
    labels_train_across_sequences = np.concatenate(labels_train_across_sequences, axis=0)
    groups = np.concatenate(groups, axis=0)

    folds = create_folds(net_act_train_across_sequences.T, labels_train_across_sequences,
                 10, groups)
    model = RidgeCV(alphas=alphas, cv=folds)
    model.fit(net_act_train_across_sequences.T, labels_train_across_sequences)

    prediction_steps_across_sequences = []

    # validation
    for i in range(n_sequences_validation):
        data = datasets.mackey_glass(n_validation + warmup, tau=random_tau, n=random_exp,
                                     x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
        start_input_val = data[warmup]
        labels_val = data[warmup + 1:]
        sim.warmup(data[:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val
        label_variance = np.var(labels_val)
        error_hist = []
        while error <= error_margin and j <= max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(np.expand_dims(output, 0))
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            error_hist.append(error)
            j += 1

        prediction_steps_across_sequences.append(j)

    return aggregate(prediction_steps_across_sequences), model, network

def eval_candidate_custom_data_signal_gen(network, unsupervised_sequences, supervised_sequences, validation_sequences,
                                          error_margin=.1, warmup=200, alphas=[10e-8, 10e-6, 10e-4, 10e-2],
                                          seed=42, shuffle=True, warmup_overlap=False):
    assert not (warmup_overlap and shuffle), "Cannot shuffle sequences if they need to " \
                                             "overlap/continue between train and validation"
    sim = NetworkSimulator(network)
    # sim.visualize(unsupervised_sequences[0])
    if shuffle:
        random.shuffle(unsupervised_sequences)
        random.shuffle(supervised_sequences)
        random.shuffle(validation_sequences)

    if len(unsupervised_sequences) > 0 and type(network) is GMMPopulationAdaptive:
        # Unsupervised: run with synaptic plasticity on different sequences
        for seq in unsupervised_sequences:
            sim.warmup(seq[:warmup])
            sim.unsupervised(seq[warmup:])
            sim.reset()

    # Supervised: one step ahead teacher forcing with fixed weights
    net_act_train_across_sequences = []
    labels_train_across_sequences = []
    groups = []
    for i, seq in enumerate(supervised_sequences):
        input = seq[warmup:-1]
        labels = seq[warmup + 1:]
        sim.warmup(seq[:warmup])
        net_act = sim.get_network_data(input)
        sim.reset()
        net_act_train_across_sequences.append(net_act)
        labels_train_across_sequences.append(labels)
        groups.append(np.ones_like(input) * i)

    net_act_train_across_sequences = np.concatenate(net_act_train_across_sequences, axis=1)
    labels_train_across_sequences = np.concatenate(labels_train_across_sequences, axis=0)
    groups = np.concatenate(groups, axis=0)

    folds = create_folds(net_act_train_across_sequences.T, labels_train_across_sequences,
                         10, groups)
    model = RidgeCV(alphas=alphas, cv=folds)
    model.fit(net_act_train_across_sequences.T, labels_train_across_sequences)

    prediction_steps_across_sequences = []

    # validation
    for i, seq in enumerate(validation_sequences):
        if warmup_overlap:
            start_input_val = seq[0]
            labels_val = seq[1:]
            sim.warmup(supervised_sequences[i][-warmup:])
        else:
            start_input_val = seq[warmup]
            labels_val = seq[warmup + 1:]
            sim.warmup(seq[:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val[0]
        label_variance = np.var(labels_val[:, 0])
        max_it_val = len(labels_val[:, 0]) - 1
        predictions = []
        # while j <= max_it_val:
        while error <= error_margin and j <= max_it_val:
            feedback_in = np.stack([feedback_in, labels_val[j, 1]])
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            predictions.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            j += 1

        prediction_steps_across_sequences.append(j)

    return np.mean(prediction_steps_across_sequences), model


def eval_candidate_signal_gen_multiple_random_sequences_adaptive_budget(network, n_sequences_unsupervised,
                                                                 n_sequences_supervised,
                                                                 n_sequences_validation,
                                                                 n_unsupervised,
                                                                 n_supervised,
                                                                 n_validation,
                                                                 max_it_val=500, warmup=400,
                                                                 alphas=[10e-14, 10e-13, 10e-12],
                                                                 seed=42,
                                                                 tau_range=[12, 22],
                                                                 n_range=[5, 15],
                                                                 x0_range=[.5, 1.2],
                                                                 starting_budget=.01,
                                                                 activation_cost=.005,
                                                                 synapse_cost=.001,
                                                                 propagation_cost=.005
                                                                ):
    sim = NetworkSimulator(network)
    random_gen = np.random.default_rng(seed=seed)
    random_tau = np.random.uniform(tau_range[0], tau_range[1])
    random_exp = np.random.uniform(n_range[0], n_range[1])

    if n_sequences_unsupervised > 0 and type(network) is GMMPopulationAdaptive:
        # Unsupervised: run with synaptic plasticity on different sequences
        for i in range(n_sequences_unsupervised):
            data = datasets.mackey_glass(n_unsupervised + warmup, tau=random_tau, n=random_exp,
                                         x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
            sim.warmup(data[:warmup])
            sim.unsupervised(data[warmup:])
            sim.reset()

    # Supervised: one step ahead teacher forcing with fixed weights
    net_act_train_across_sequences = []
    labels_train_across_sequences = []
    groups = []
    for i in range(n_sequences_supervised):
        data = datasets.mackey_glass(n_supervised + warmup, tau=random_tau, n=random_exp,
                                     x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
        input = data[warmup:-1]
        labels = data[warmup + 1:]
        sim.warmup(data[:warmup])
        net_act = sim.get_network_data(input)
        sim.reset()
        net_act_train_across_sequences.append(net_act)
        labels_train_across_sequences.append(labels)
        groups.append(np.ones_like(input) * i)

    net_act_train_across_sequences = np.concatenate(net_act_train_across_sequences, axis=1)
    labels_train_across_sequences = np.concatenate(labels_train_across_sequences, axis=0)
    groups = np.concatenate(groups, axis=0)

    folds = create_folds(net_act_train_across_sequences.T, labels_train_across_sequences,
                 10, groups)
    model = RidgeCV(alphas=alphas, cv=folds)
    model.fit(net_act_train_across_sequences.T, labels_train_across_sequences)

    prediction_steps_across_sequences = []
    leftover_budget_across_sequences = []
    budget = starting_budget
    # validation
    for i in range(n_sequences_validation):
        data = datasets.mackey_glass(n_validation + warmup, tau=random_tau, n=random_exp,
                                     x0=np.random.uniform(x0_range[0], x0_range[1]), seed=random_gen)
        start_input_val = data[warmup]
        labels_val = data[warmup + 1:]
        sim.warmup(data[:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val
        label_variance = np.var(labels_val)

        while budget > 0 and j <= max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            power = network.get_current_power_bio(activation_cost, synapse_cost, propagation_cost)
            budget -= power * network.dt
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            budget += energy_reward(error, scale=.02)
            j += 1

        leftover_budget_across_sequences.append(np.clip(budget, a_min=0, a_max=None))
        prediction_steps_across_sequences.append(j)

    return np.mean(prediction_steps_across_sequences), model, np.mean(leftover_budget_across_sequences)

def energy_reward(error, scale=.05, shape=60):
    return scale * np.exp(-1 * shape * error)

def single_sample_NRSE(prediction, target, variance):
    error = mse(target, prediction)
    error = np.sqrt(error) / variance
    return error


def eval_candidate(network, train_data, val_data, warmup=400, lag=0):
    """
    Evaluates NRMSE of a DistDelayNetwork

    :param network: DistDelayNetwork
        candidate
    :param data: ndarray
        2 by N array containing input and labels
    :param warmup: int
        Warmup drop
    :return: (train NRMSE score, validation NRMSE score)
    """
    train_input = train_data[0, warmup:]
    train_input_warmup = train_data[:warmup]
    train_labels = train_data[1, warmup:]
    sim = NetworkSimulator(network)

    # run warmup
    sim.warmup(train_input_warmup)

    # generate net activity
    train_net_act = sim.get_network_data(train_input).T
    model = Ridge(alpha=0)

    if lag > 0:
        model.fit(train_net_act[lag:], train_labels[:-lag])
    elif lag == 0:
        model.fit(train_net_act, train_labels)
    else:
        model.fit(train_net_act[:lag], train_labels[-lag:])

    train_predictions = model.predict(train_net_act)
    if lag > 0:
        train_performance = nrmse(train_predictions[lag:], train_labels[:-lag])
    elif lag == 0:
        train_performance = nrmse(train_predictions, train_labels)
    else:
        train_performance = nrmse(train_predictions[:lag], train_labels[-lag:])

    # test readout
    sim.reset()
    val_input = val_data[0, :]
    val_labels = val_data[1, warmup:]
    val_net_act = sim.get_network_data(val_input).T
    val_predictions = model.predict(val_net_act)
    if lag > 0:
        val_performance = nrmse(val_predictions[lag:], val_labels[:-lag])
    elif lag == 0:
        val_performance = nrmse(val_predictions, val_labels)
    else:
        val_performance = nrmse(val_predictions[:lag], val_labels[-lag:])

    return train_performance, val_performance


def plot_learning_curve(network, train_data, val_data, warmup=400, lag=0, ax=plt):
    train_input = train_data[0, :]
    train_labels = train_data[1, warmup:]
    val_input = val_data[0, :]
    val_labels = val_data[1, warmup:]
    sim = NetworkSimulator(network, warmup=warmup)

    # generate net activity
    train_net_act = sim.get_network_data(train_input).T
    sim.reset()
    val_net_act = sim.get_network_data(val_input).T

    N = train_data.shape[1]
    n_ticks = np.array(N * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), dtype='int64')
    train_perf = []
    val_perf = []
    for n in n_ticks:
        # train perf
        model = Ridge(alpha=0)

        if lag > 0:
            model.fit(train_net_act[lag:n], train_labels[:n - lag])
        elif lag == 0:
            model.fit(train_net_act[:n], train_labels[:n])
        else:
            return None

        train_predictions = model.predict(train_net_act)
        if lag > 0:
            train_performance = nrmse(train_predictions[lag:n], train_labels[:n - lag])
        elif lag == 0:
            train_performance = nrmse(train_predictions[:n], train_labels[:n])

        # val perf
        val_predictions = model.predict(val_net_act)

        if lag > 0:
            val_performance = nrmse(val_predictions[lag:], val_labels[:-lag])
        elif lag == 0:
            val_performance = nrmse(val_predictions, val_labels)

        train_perf.append(train_performance)
        val_perf.append(val_performance)

    ax.plot(n_ticks, train_perf)
    ax.plot(n_ticks, val_perf)
    ax.set_xlabel('N')
    ax.set_ylabel('NRMSE')


def plot_learning_curve_signal_gen_supervised(network, unsupervised_data, train_data, val_data, warmup=400,
                                   alphas=[10e-14, 10e-13, 10e-12]):

    sim = NetworkSimulator(network)

    # Unsupervised: run with synaptic plasticity
    warmup_unsupervised = unsupervised_data[:warmup]
    input_unsupervised = unsupervised_data[warmup:]
    sim.warmup(warmup_unsupervised)
    sim.unsupervised(input_unsupervised)

    # Training: train with one step ahead prediction
    warmup_train = train_data[:warmup]
    input_train = train_data[warmup: -1]
    labels_train = train_data[warmup + 1:]
    sim.warmup(warmup_train)
    net_act_train = sim.get_network_data(input_train).T
    sim.reset()

    # label_variance = np.var(np.concatenate([unsupervised_data, train_data, val_data]))
    # val data
    warmup_val = val_data[:warmup]
    input_val = val_data[warmup: -1]
    labels_val = val_data[warmup + 1:]

    N = train_data.shape[0] - warmup
    n_ticks = np.array(N * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), dtype='int64')
    train_perf = []
    val_perf = []
    for n in n_ticks:
        # teacher forcing one step ahead
        net_act_train_limited = net_act_train[:n, :]
        labels_train_limited = labels_train[:n]
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(net_act_train_limited, labels_train_limited)

        # Performance on train set
        # fitness measure is amount of prediction steps within error margin
        start_input_val = input_train[0]
        sim.reset()
        sim.warmup(warmup_train)

        feedback_in = start_input_val
        error = 0
        i = 0
        label_variance = np.var(train_data)
        error_margin = .1
        max_it_val = n
        error_sum = 0
        predictions = []
        while error <= error_margin and i <= max_it_val:
        # while i < max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            predictions.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_train_limited[i, 0],
                                       label_variance)  # np.abs(labels_val[i, 0] - feedback_in)
            error_sum += error
            i += 1


        train_perf.append(i)
        # train_perf.append(nrmse(predictions, labels_train_limited[:, 0].T))
        sim.reset()

        # Performance on validation set
        start_input_val = input_val[0]
        sim.warmup(warmup_val)

        feedback_in = start_input_val
        error = 0
        j = 0
        label_variance = np.var(val_data)
        error_margin = .1
        max_it_val = len(labels_val)
        error_sum = 0

        predictions = []
        while error <= error_margin and j <= max_it_val:
        # while j < max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            predictions.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)  # np.abs(labels_val[i, 0] - feedback_in)
            j += 1
            # error_sum += error
        val_perf.append(j)
        # val_perf.append(nrmse(predictions, labels_val[:, 0].T))
        sim.reset()

    plt.plot(n_ticks, train_perf)
    plt.plot(n_ticks, val_perf)
    plt.xlabel('N')
    plt.ylabel('NRMSE')
    plt.legend(['train', 'validation'])
    plt.title('Varying number of supervised samples, N_unsupervised = ' + str(len(unsupervised_data)))
    plt.show()

def plot_learning_curve_signal_gen_unsupervised(network, unsupervised_data, train_data, val_data, warmup=400,
                                   alphas=[10e-14, 10e-13, 10e-12]):

    sim = NetworkSimulator(network)

    # Unsupervised: run with synaptic plasticity
    warmup_unsupervised = unsupervised_data[:warmup]
    input_unsupervised = unsupervised_data[warmup:]
    sim.warmup(warmup_unsupervised)

    N = unsupervised_data.shape[0] - warmup
    n_ticks = np.array(N * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), dtype='int64')
    train_perf = []
    val_perf = []
    for n in n_ticks:
        input_unsupervised_limited = input_unsupervised[:n]
        sim.unsupervised(input_unsupervised_limited)

        # Training: train with one step ahead prediction
        warmup_train = train_data[:warmup]
        input_train = train_data[warmup: -1]
        labels_train = train_data[warmup + 1:]
        sim.warmup(warmup_train)
        net_act_train = sim.get_network_data(input_train).T
        sim.reset()

        # label_variance = np.var(np.concatenate([unsupervised_data, train_data, val_data]))
        # val data
        warmup_val = val_data[:warmup]
        input_val = val_data[warmup: -1]
        labels_val = val_data[warmup + 1:]

        # teacher forcing one step ahead
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(net_act_train, labels_train)

        # Performance on train set
        # fitness measure is amount of prediction steps within error margin
        start_input_val = input_train[0]
        sim.reset()
        sim.warmup(warmup_train)
        feedback_in = start_input_val
        i = 0
        error_margin = .1
        max_it_val = len(labels_train)
        predictions = []
        error = 0
        label_variance = np.var(labels_train)
        while error <= error_margin and i <= max_it_val:
        # while i < max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            predictions.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_train[i, 0],
                                       label_variance)  # np.abs(labels_val[i, 0] - feedback_in)
            # error_sum += error
            i += 1


        train_perf.append(i)
        # train_perf.append(nrmse(predictions, labels_train[:, 0].T))
        sim.reset()

        # Performance on validation set
        start_input_val = input_val[0]
        sim.warmup(warmup_val)

        feedback_in = start_input_val
        error = 0
        j = 0
        label_variance = np.var(val_data)
        error_margin = .1
        max_it_val = len(labels_val)
        error_sum = 0

        predictions = []
        while error <= error_margin and j <= max_it_val:
        # while j < max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            predictions.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)  # np.abs(labels_val[i, 0] - feedback_in)
            j += 1
            # error_sum += error
        # val_perf.append(nrmse(predictions, labels_val[:, 0].T))
        val_perf.append(j)
        sim.reset()

    plt.plot(n_ticks, train_perf)
    plt.plot(n_ticks, val_perf)
    plt.xlabel('N')
    plt.ylabel('Prediction steps')
    plt.legend(['train', 'validation'])
    plt.title('Varying number of unsupervised samples, N_supervised = ' + str(len(train_data)))
    plt.show()

def load_from_saved_ES(path, which_net=0):
    assert which_net in [0, 1, 2], 'Parameter that selects which network should be 0, 1, or 2, for best network ever, \
    best network from best average generation, or best network from last generation respectively'

    data = pkl.load(open(path, "rb"))
    empty_net = data['example net']


    net_params = data['evolutionary strategy'].best.x
    if which_net == 1:
        parameters = data['parameters']
        val_performance = data['validation performance']
        val_performance = np.mean(val_performance, axis=-1)
        average_val_per_gen = np.mean(val_performance, axis=-1)
        best_gen = np.argmin(average_val_per_gen)
        best_ind = np.argmin(val_performance[best_gen])
        net_params = parameters[best_gen, best_ind]

    new_net = empty_net.get_new_network_from_serialized(net_params)
    return new_net


def load_from_saved_ES_specific_gen(path, gen, ind=None):
    data = pkl.load(open(path, "rb"))
    empty_net = data['example net']
    parameters = data['parameters']
    val_performance = data['validation performance']
    val_performance = np.mean(val_performance, axis=-1)

    if ind is None:
        best_ind = np.argmin(val_performance[gen])
        ind = best_ind

    net_params = parameters[gen, ind]
    new_net = empty_net.get_new_network_from_serialized(net_params)
    return new_net


def genome_memory_capacity(hyperparameters, start_net, max_delay, sequence_length, z_function=None, warmup_time=400,
                           alphas=[1, 10, 100], genome_reps=5, eval_reps=5, n_sequences_unsupervised=5,
                           n_unsupervised=500, n_range=(10, 10), tau_range=(17,17), seed=42):
    total_m_cap = []
    for i in range(genome_reps):
        net = start_net.get_new_network_from_serialized(hyperparameters)
        bcm_sim = NetworkSimulator(net)
        random_tau = np.random.uniform(tau_range[0], tau_range[1])
        random_exp = np.random.uniform(n_range[0], n_range[1])
        random_gen = np.random.default_rng(seed=seed)

        if n_sequences_unsupervised > 0 and type(net) is GMMPopulationAdaptive:
            # Unsupervised: run with synaptic plasticity on different sequences
            print("Training BCM connections")
            for i in range(n_sequences_unsupervised):
                data = datasets.mackey_glass(n_unsupervised + warmup_time, tau=random_tau, n=random_exp,
                                             x0=np.random.uniform(.5, 1.2), seed=random_gen)
                bcm_sim.warmup(data[:warmup_time])
                bcm_sim.unsupervised(data[warmup_time:])
                bcm_sim.reset()

        m_cap = network_memory_capacity(net, max_delay, sequence_length, z_function, warmup_time, alphas, eval_reps)
        total_m_cap.append(m_cap)

    return total_m_cap


def genome_memory_capacity_evolvable(hyperparameters, start_net, max_delay, sequence_length, z_function=None, warmup_time=400,
                           alphas=[1, 10, 100], genome_reps=5, eval_reps=5):
    total_m_cap = []
    for i in range(genome_reps):
        net = start_net.get_new_evolvable_population_from_serialized(hyperparameters)
        m_cap = network_memory_capacity(net, max_delay, sequence_length, z_function, warmup_time, alphas, eval_reps)
        total_m_cap.append(m_cap)

    return total_m_cap


def network_memory_capacity(network, max_delay, sequence_length, z_function=None, warmup_time=400, alphas=[1, 10, 100], reps=5):
    total_m_cap = []
    for i in range(reps):
        m_cap = memory_capacity(network, max_delay, sequence_length, z_function, warmup_time, alphas)
        total_m_cap.append(m_cap)

    return np.mean(total_m_cap, axis=0)


def memory_capacity(network, max_delay, sequence_length, z_function=None, warmup_time=400, alphas=[1, 10, 100]):
    noise_input_train = np.random.uniform(size=sequence_length + warmup_time)
    z_output_train = np.copy(noise_input_train)
    if z_function is not None:
        z_output_train = z_function(noise_input_train)

    # Training
    sim = NetworkSimulator(network)
    train_net_act = sim.get_network_data(noise_input_train)

    readout_list = []
    for d in range(max_delay):
        readout = RidgeCV(alphas=alphas, cv=5)
        readout.fit(train_net_act[:, warmup_time:].T, z_output_train[warmup_time-d:len(z_output_train)-d])
        readout_list.append(readout)

    # Testing
    sim.reset()
    noise_input_test = np.random.uniform(size=sequence_length + warmup_time)
    z_output_test = np.copy(noise_input_test)
    if z_function is not None:
        z_output_test = z_function(noise_input_test)

    test_net_act = sim.get_network_data(noise_input_test)

    m_caps = []

    for d, readout in enumerate(readout_list):
        predictions = readout.predict(test_net_act[:, warmup_time:].T)
        MC = np.corrcoef(predictions, z_output_test[warmup_time-d:len(z_output_test)-d])[0,1]
        m_caps.append(MC)

    return m_caps


def act_dimensionality(network_activity, variance_threshold=.95):
    _, s, _ = np.linalg.svd(network_activity)
    dim = 0
    var_explained = 0
    while var_explained < variance_threshold:
        var_explained = np.sum(s[:dim + 1]) / np.sum(s)
        dim += 1
    return dim

def genotype_dimensionality(network, net_input, measuring_reps, variance_threshold=.95, warmup=400):
    dims = []
    serialized_parameters = network.get_serialized_network_parameters()

    for rep in range(measuring_reps):
        # resample_network
        network = network.get_new_evolvable_population_from_serialized(serialized_parameters)
        sim = NetworkSimulator(network)
        sim.warmup(net_input[:warmup])
        net_act = sim.get_network_data(net_input[warmup:])
        dim = act_dimensionality(net_act, variance_threshold=variance_threshold)
        dims.append(dim)

    return dims

# def baseline2DDN(baseline_net):
#     k = evolved_bl_net.k
#
#     mix = np.ones((k,))
#     mix = softmax(mix)
#
#     var = np.ones((k, 2)) * 0.1
#     corr = np.ones((k,)) * 0
#     mu = np.zeros((k, 2))
#
#     added_delays_net = populations.GMMPopulation(evolved_bl_net.N, mix, mu, var, corr, evolved_bl_net.inhibitory,
#                                                  evolved_bl_net.connectivity, evolved_bl_net.cluster_connectivity,
#                                                  evolved_bl_net.weight_scaling, evolved_bl_net.weight_mean,
#                                                  evolved_bl_net.bias_scaling, evolved_bl_net.bias_mean,
#                                                  evolved_bl_net.decay, evolved_bl_net.size_in, evolved_bl_net.size_out,
#                                                  (0, 0), evolved_bl_net.activation_func, evolved_bl_net.autapse,
#                                                  .0025, evolved_bl_net.x_lim, evolved_bl_net.y_lim,
#                                                  evolved_bl_net.fixed_delay)