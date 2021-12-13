import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from numpy.random import default_rng
from scipy.stats import (
    spearmanr,
    pearsonr,
)
from scipy.special import gamma
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
    # scale noise to be between zero and one
    data = np.interp(data, (data.min(), data.max()), (-1, +1))
    if weights is None:
        weights = rng.uniform(0, 1, s)
    # add signal to noise.
    data += (weights * signal[:, None]).T
    return data, weights


def make_sine_signal(frequency=5,
                     theta=0,
                     amplitude=1,
                     stype='sine'):
    """
    Generate different sine wave signals with a given phase offset
    :param frequency: float, signal frequency
    :param theta: float, phase offset of the signal
    :param amplitude: float, amplitude of the signal
    :return:
    """
    # noise is drawn from a standard normal distribution.
    print(f"Generating a signal with a frequency of {frequency}, an "
          f"amplitude of {amplitude}, and a phase shift of {theta} samples")
    # generate a sine wave with known properties
    data_size = 10000
    zeros = np.zeros(data_size)
    signal_size = 1000
    x = np.arange(0, 1, 1 / signal_size)
    if stype == 'sine':
        # make a sine wave
        signal = amplitude * np.sin(2 * np.pi * frequency * x)
    else:
        # make something more interesting
        phase = 7
        srate = 1000
        from scipy.signal import tukey
        signal = np.sin(phase*2*np.pi + 2*np.pi*frequency*x)
        modLatency = 170
        modWidth= 700
        latency = np.floor((modLatency/1000)*srate) + 1
        taper = 10
        width = int(np.floor((modWidth/1000)*srate))
        win = tukey(width, taper)
        #win[0:int((np.ceil(width / 2) - latency))] = 0
        w = np.zeros(len(signal))
        w[200:200+len(win)] = win
        signal = signal * w
    # scale signal to be between zero and one
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, +1))
    zeros[theta:theta+signal_size] = signal
    fig = plt.plot(zeros)
    # save the plot
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


def plot_srm(model,
             weights,
             space='spectral',
             outdir=None):
    """
    Plot the components of a shared response model as well as its weights
    :param model: SRM model
    :param weights: specific weights
    :param space: str; description of the model space, used in names/titles
    :param outdir: str, path to existing directory.
    If given, plots will be saved in this directory
    :return:
    """
    # plot of the components
    ax = sns.lineplot(data=[model.s_[i] for i in range(model.s_.shape[0])],
                      linewidth=1)
    ax.set(xlabel='sample frequencies' if space == 'spectral' else 'samples',
           ylabel='a.U.',
           title=f'Components in shared ({space}) space')
    if outdir is not None:
        outpath = Path(outdir) / f'components_{space}-space_ds.svg'
        ax.figure.savefig(outpath)

    # individual plots
    for i in range(len(weights)):
        n_components = model.w_[i].shape[1]
        # write weights from all components into a consecutive list
        model_weights = [model.w_[i][:, c] for c in range(n_components)]
        model_weights = np.asarray(
            [item for sublist in model_weights for item in sublist]
        )
        data = {'component': np.repeat(range(n_components), len(weights[i])),
                'ground truth': np.tile(weights[i], n_components),
                'model weights': model_weights
                }

        g = sns.JointGrid(data=data,
                          hue='component',
                          x='ground truth',
                          y='model weights',
                          ylim=[-0.3, 0.3]
                          )
        g.plot(sns.scatterplot, sns.histplot, alpha=.6)
        g.fig.suptitle('Relationship between model weights \n'
                       'and ground truth weights for each component',
                       verticalalignment='baseline')
        if outdir is not None:
            # save the plot
            outpath = Path(outdir) / \
                      f'model-weights_versus_ground-truth_ds-{i}.svg'
            g.fig.savefig(outpath, bbox_inches='tight')

    # one plot with all data
    components = np.tile(np.repeat(range(n_components), len(weights[0])), len(weights))
    ground_truth = [np.tile(w, n_components) for w in weights]
    ground_truth = np.asarray(
        [item for sublist in ground_truth for item in sublist]
    )
    model_weights = []
    for w in model.w_:
        model_weight = np.concatenate([w[:, c] for c in range(n_components)])
        model_weights.extend(model_weight)
    model_weights = np.asarray(model_weights)
    data = {'component': components,
             'ground truth': ground_truth,
             'model weights': model_weights
           }
    g = sns.JointGrid(data=data,
                      hue='component',
                      x='ground truth',
                      y='model weights',
                      ylim=[-0.3, 0.3]
                      )
    g.plot(sns.scatterplot, sns.histplot, alpha=.1)
    g.fig.suptitle('Relationship between model weights \n'
                   'and ground truth weights for each component',
                   verticalalignment='baseline')
    if outdir is not None:
        # save the plot
        outpath = Path(outdir) / \
                  f'model-weights_versus_ground-truth_all-ds.svg'
        g.fig.savefig(outpath, bbox_inches='tight')


def simulate(n=5):
    """
    Simulate data
    :param n: int, number of subjects to simulate
    :return:
    """
    # make data without any offset:
    simulated_data = [simulate_raw(make_sine_signal(stype='else')) for i in range(15)]
    raw, weights = list(zip(*simulated_data))
    model = fit_srm(raw)
    plot_srm(model, weights, space='time-resolved', outdir='/tmp')
    # make data with offset
    simulated_data = [simulate_raw(make_sine_signal(stype='else', theta=i))
                      for i in rng.uniform(0, 9000, 15).astype(int)]
    raw, weights = list(zip(*simulated_data))
    model = fit_srm(raw)
    plot_srm(model, weights, space='time-resolved', outdir='/tmp')
    # make data with offset, but transform into power spectrum before SRM
    simulated_data = [simulate_raw(make_sine_signal(stype='else', theta=i))
                      for i in rng.uniform(0, 9000, 15).astype(int)]
    raw, weights = list(zip(*simulated_data))
    power = transform_to_power(raw)
    model = fit_srm(power)
    plot_srm(model, weights, space='spectral', outdir='/tmp')

    # Now, use identical weights for each subject. Check whether the
    # transformation (model.w_) obtained with a phase-locked signal and the
    # transformation obtained with the same signal with random phase
    # shifts are comparable.

    # make subject weights:
    subject_weights = [rng.uniform(0, 1, 306) for i in range(15)]
    # get model transformations for data with no phase shift
    simulated_data = [simulate_raw(make_sine_signal(stype='else', theta=100), weights=weight)
                      for weight in subject_weights]
    raw, weights = list(zip(*simulated_data))
    model_time = fit_srm(raw)
    plot_srm(model_time, weights, space='time-resolved', outdir='/tmp')

    # get model transformation for phase-shifted data in power spectrum
    simulated_data = [simulate_raw(make_sine_signal(stype='else', theta=theta), weights=subject_weights[i])
                      for i, theta in enumerate(rng.uniform(0, 9000, 15).astype(int))]
    raw, weights = list(zip(*simulated_data))
    power = transform_to_power(raw)
    model_spect = fit_srm(power)
    plot_srm(model_spect, weights,space='spectral', outdir='/tmp')
    # check how similar the weights for each feature are
    [spearmanr(model_time.w_[i][:, 0], model_spect.w_[i][:, 0])
     for i in range(15)]
    [spearmanr(subject_weights[i], model_spect.w_[i][:, 0])
     for i in range(15)]
    [spearmanr(model_time.w_[i][:, 1], model_spect.w_[i][:, 1])
     for i in range(15)]
    [spearmanr(subject_weights[i], model_spect.w_[i][:, 1])
     for i in range(15)]

    # transform the raw data into the shared space using the model weights, and plot it together with the
    # generated signal
    transformed = np.dot(model_spect.w_[0].T, raw[0])[0]
    plt.plot(transformed)


    # TODO: take subject trials and use the model transformations to obtain
    #  component weights in the time domain for (hopefully) interpretable
    #  SRM components

    # TODO: take response time, not delay. Check whether there is a correlation
    #  between response time and the (hopefully to be found) SRM component of
    #  interest