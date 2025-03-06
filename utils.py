import numpy as np
from simulator import NetworkSimulator
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from reservoirpy import datasets
from populations import GMMPopulationAdaptive


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

    val_performance_per_lag = np.zeros_like(lag_grid, dtype='float64')
    # TODO: replace with list to avoid weird averages
    train_performance_per_lag = np.zeros_like(lag_grid, dtype='float64')
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

        val_performance_per_lag[lag] = val_performance
        train_performance_per_lag[lag] = train_performance
        model_per_lag[lag] = model

    return train_performance_per_lag, val_performance_per_lag, model_per_lag


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


def eval_candidate_signal_gen_multiple_random_sequences_adaptive(network, n_sequences_unsupervised,
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
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            feedback_in = model.predict(output)[0][0]
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            j += 1

        prediction_steps_across_sequences.append(j)
    prediction_step = None
    if len(prediction_steps_across_sequences) > 0:
        prediction_step = np.mean(prediction_steps_across_sequences)
    return prediction_step, model, network

def single_sample_NRSE(prediction, target, variance):
    error = mse(target, prediction)
    error = np.sqrt(error) / variance
    return error
