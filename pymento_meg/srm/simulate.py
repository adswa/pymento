import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import spearmanr
from brainiak.funcalign import srm

rng = default_rng()

def simulate_raw(signal,
                 s=306,
                 weights=None,
                 ):
    """
    Generate simulated data for s sensors from a signal with known properties by
    adding random noise, and a sensor-specific weighted ground-truth signal.
    Weights are random.
    :param signal: a generated pure signal
    :param s: Number of sensors
    :return: data, weights
    """

    # this is pure noise
    data = rng.standard_normal(len(signal) * s).reshape((s, -1))
    if weights is None:
        weights = rng.uniform(0, 1, s)
    # add signal to noise.
    data += (weights * signal[:, None]).T
    return data, weights


def make_sine_signal(frequency=5,
                     theta=0):
    """
    Generate different sine wave signals with a given phase offset
    :param frequency: float, signal frequency
    :param theta: float, phase offset of the signal
    :return:
    """
    # generate a sine wave with known properties
    data_size = 10000
    zeros = np.zeros(data_size)
    signal_size = 1000
    x = np.arange(0, 1, 1 / signal_size)
    amplitude = 1

    signal = amplitude * np.sin(2 * np.pi * frequency * x)
    zeros[theta:theta+signal_size] = signal
    return zeros


def transform_to_power(signal):
    """
    Transform a signal into a power spectrum
    :return:
    """
    power = np.abs(np.fft.fft(signal)) ** 2
    return power


def fit_srm(data,
            features=2):
    """
    Fit a shared response model on the data
    :param data: list of arrays, the simulated data
    :param features: int, number of features to use
    :return:
    """
    model = srm.SRM(features=features)
    model.fit(data)
    return model


def plot_srm(model, weights):
    """
    Plot the components of a shared response model
    :param model: SRM model
    :param weights: specific weights
    :return:
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(model.s_[0])
    ax1.plot(model.s_[1])
    fig2, ax2 = plt.subplots()
    ax2.scatter(weights[0], model.w_[0][:, 1])
    ax2.scatter(weights[0], model.w_[0][:, 0])


def simulate(n=5):
    """
    Simulate data
    :param n: int, number of subjects to simulate
    :return:
    """
    # make data without any offset:
    simulated_data = [simulate_raw(make_sine_signal()) for i in range(15)]
    raw, weights = list(zip(*simulated_data))
    model = fit_srm(raw)
    plot_srm(model, weights)
    # make data with offset
    simulated_data = [simulate_raw(make_sine_signal(theta=i))
                      for i in rng.uniform(0, 9000, 15).astype(int)]
    raw, weights = list(zip(*simulated_data))
    model = fit_srm(raw)
    plot_srm(model, weights)
    # make data with offset, but transform into power spectrum before SRM
    simulated_data = [simulate_raw(make_sine_signal(theta=i))
                      for i in rng.uniform(0, 9000, 15).astype(int)]
    raw, weights = list(zip(*simulated_data))
    power = transform_to_power(raw)
    model = fit_srm(power)
    plot_srm(model, weights)

    # Now, use identical weights for each subject. Check whether the
    # transformation (model.w_) obtained with a phase-locked signal and the
    # transformation obtained with the same signal with random phase
    # shifts are comparable.

    # make subject weights:
    subject_weights = [rng.uniform(0, 1, 306) for i in range(15)]
    # get model transformations for data with no phase shift
    simulated_data = [simulate_raw(make_sine_signal(theta=100), weights=weight)
                      for weight in subject_weights]
    raw, weights = list(zip(*simulated_data))
    model_time = fit_srm(raw)
    plot_srm(model_time, weights)

    # get model transformation for phase-shifted data in power spectrum
    simulated_data = [simulate_raw(make_sine_signal(theta=theta), weights=subject_weights[i])
                      for i, theta in enumerate(rng.uniform(0, 9000, 15).astype(int))]
    raw, weights = list(zip(*simulated_data))
    power = transform_to_power(raw)
    model_spect = fit_srm(power)
    plot_srm(model_spect, weights)
    # check how similar the weights for each feature are
    [spearmanr(model_time.w_[i][:, 0], model_spect.w_[i][:, 0])
     for i in range(15)]
    [spearmanr(model_time.w_[i][:, 1], model_spect.w_[i][:, 1])
     for i in range(15)]

