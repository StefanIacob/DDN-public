import numpy as np
from simulator import NetworkSimulator
from scipy.stats import norm
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt


def mse(target_signal,input_signal):
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


def nmse(target_signal,input_signal):
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
    var = np.std(target_signal)** 2

    return mse(target_signal,input_signal) / var

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

    return np.sqrt(nmse(target_signal,input_signal))


def state_entropy(A):
    N = len(A)
    std = np.std(A)
    state_kernel = []
    for i in range(N):
        for j in range(N):
            kern = norm.pdf(A[i]-A[i], 0, 0.3*std)
            state_kernel.append(kern)
    H2 = -1 * np.log((1/N**2) * np.sum(state_kernel))
    return H2


def spectral_radius_norm(W, wanted_sr):
    v = np.linalg.eigvals(W)
    sr = np.max(np.absolute(v))
    W_scaled = (W * wanted_sr)/sr
    return W_scaled


def createNARMA(length=10000, system_order=10, coef = [.3, .05, 1.5, .1]):

    inputs = np.random.rand(length, 1) * .5
    inputs.shape = (-1, 1)
    outputs = np.zeros((length, 1))
    for k in range(system_order - 1, length - 1):
        outputs[k + 1] = coef[0] * outputs[k] + coef[1] * \
                                           outputs[k] * np.sum(outputs[k - (system_order - 1):k + 1]) + \
                         coef[2] * inputs[k - (system_order-1)] * inputs[k] + coef[3]
    return inputs, outputs


def createNARMA10(length=10000):
    return createNARMA(length=length, system_order=10, coef=[.3, .05, 1.5, .1])


def createNARMA30(length=10000):
    return createNARMA(length=length, system_order=30, coef=[.2, .04, 1.5, .001])


def eval_candidate_lag_gridsearch(network, train_data, val_data, warmup=400,
                                  lag_grid=range(0, 15), alphas=[10e-14, 10e-13, 10e-12]):
    assert np.all(np.array(lag_grid) >= 0), 'No negative lag allowed'
    train_input = train_data[0, :]
    train_labels = train_data[1, warmup:]
    val_input = val_data[0, :]
    val_labels = val_data[1, warmup:]
    sim = NetworkSimulator(network, warmup=warmup)

    # generate net activity
    train_net_act = sim.get_network_data(train_input).T
    sim.reset()
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
    train_input = train_data[0, :]
    train_labels = train_data[1, warmup:]
    sim = NetworkSimulator(network, warmup=warmup)

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
            model.fit(train_net_act[lag:n], train_labels[:n-lag])
        elif lag == 0:
            model.fit(train_net_act[:n], train_labels[:n])
        else:
            return None

        train_predictions = model.predict(train_net_act)
        if lag > 0:
            train_performance = nrmse(train_predictions[lag:n], train_labels[:n-lag])
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