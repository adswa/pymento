import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from numpy.random import default_rng
from scipy.stats import (
    spearmanr,
)
from scipy.signal import tukey
from brainiak.funcalign import srm

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARN)

rng = default_rng()

def simulate_raw(signal,
                 s=306,
                 weights=None,
                 s_without_signal=0
                 ):
    """
    Generate simulated data for s sensors from a signal with known properties by
    adding random noise, and a sensor-specific weighted ground-truth signal.
    Weights are random.
    If set to an integer, s_with_signal only lets the specified amount of
    sensors have signal.
    :param signal: a generated pure signal
    :param s: Number of sensors
    :param s_without_signal: float, percentage of sensors without signal
    :param weights: list, pre-existing weights to use. Cave, when pre-existing
    weights are used channels will not be set to zero or normalized
    :return: data, weights, signal
    """

    # this is pure noise
    data = rng.standard_normal(len(signal) * s).reshape((s, -1))
    # scale noise to be between zero and one
    data = np.interp(data, (data.min(), data.max()), (-1, +1))
    if weights is None:
        # generate new, random weights, and set the specified percentage of them
        # randomly to zero
        weights = rng.uniform(0, 1, s)
        assert 0 <= s_without_signal <= 1
        logging.debug(
            f"Setting {s_without_signal*100}% of channels to 0 signal"
        )
        zeroindices = np.random.choice(np.arange(weights.size), replace=False,
                                       size=int(weights.size * s_without_signal)
                                       )
        weights[zeroindices] = 0
        # scale weights to a sum of 1 to ensure identical scaling across runs
        weights[~zeroindices] / np.sum(weights[~zeroindices])
    # add signal to noise.
    data += (weights * signal[:, None]).T
    fig, ax = plt.subplots()
    ax.plot(data.T, linewidth=1)
    ax.set(xlabel='samples',
           ylabel='amplitude',
           title=f'Artificial signal embedded in noise')
    plt.tight_layout()
    plt.close()
    return data, weights, [signal]


def make_signal(frequency=10,
                theta=0,
                amplitude=1,
                stype='sine',
                data_size=10000,
                signal_size=1000,
                outdir='/tmp',
                nsignal=1,
                ):
    """
    Generate different signals with a given phase offset
    :param frequency: float, signal frequency
    :param theta: float, phase offset of the signal
    :param amplitude: float, amplitude of the signal
    :param data_size: int, length of the total time series
    :param signal_size: int, length of signal embedded in the time series
    :param nsignal: int, number of waves in the signal
    :return:
    """
    # noise is drawn from a standard normal distribution.
    logging.debug(f"Generating a signal with a frequency of {frequency}, an "
                 f"amplitude of {amplitude}, and a phase shift of {theta} "
                 f"samples")
    # generate a sine wave with known properties
    timeseries = np.zeros(data_size)
    x = np.arange(0, 1, 1 / signal_size)
    for i in range(nsignal):
        if stype == 'sine':
            # make a sine wave
            signal = amplitude * np.sin(2 * np.pi * frequency * x)
        else:
            # make something more interesting
            phase = 7
            srate = 1000
            signal = np.sin(phase*2*np.pi + 2*np.pi*(frequency+i*3)*x)
            modWidth= 700
            taper = 10
            width = int(np.floor((modWidth/1000)*srate))
            win = tukey(width, taper)
            w = np.zeros(len(signal))
            w[200:200+len(win)] = win
            signal = signal * w
        # scale signal to be between zero and one
        signal = np.interp(signal, (signal.min(), signal.max()), (-1, +1))
        if i != 0:
            theta += i*1000
            if theta+signal_size > len(timeseries):
                theta = theta+signal_size - len(timeseries) + 333
        timeseries[theta:theta+signal_size] += signal
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=np.arange(0, data_size), y=timeseries, linewidth=1)
    ax.set(xlabel='samples',
           ylabel='amplitude',
           title=f'Artificially generated signal component ({frequency}Hz)')
    ax.plot(data=timeseries)

    # save the plot
    outpath = Path(outdir) / \
              f'signal.svg'
    ax.figure.savefig(outpath, bbox_inches='tight')
    plt.close()
    return timeseries


def transform_to_power(signal):
    """
    Transform a signal into a power spectrum
    :return:
    """
    power = np.abs(np.fft.rfft(signal))
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


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_srm(model,
             weights,
             space='spectral',
             outdir='/tmp'):
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
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=[model.s_[i] for i in range(model.s_.shape[0])],
                      linewidth=1)
    ax.set(xlabel='sample frequencies' if space == 'spectral' else 'samples',
           ylabel='a.U.',
           title=f'Components in shared ({space}) space')
    outpath = Path(outdir) / f'components_{space}-space_ds.svg'
    ax.figure.savefig(outpath)

    # big potpourri of individual plots
    total_plots = len(weights)
    columns = 3 if total_plots < 20 else 5
    # Compute Rows required
    rows = total_plots // columns
    rows += total_plots % columns
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(columns, rows)
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
                          ylim=[-0.3, 0.3],
                          )
        m = SeabornFig2Grid(g, fig, gs[i])
        g.plot(sns.scatterplot, sns.histplot, alpha=.5)
    m.fig.tight_layout()
    # save the plot
    outpath = Path(outdir) / \
              f'model-weights_versus_ground-truth_ds-individual.svg'
    g.fig.suptitle('Model weights (left) vs ground truth weights '
                   '(bottom) for each component',
                   verticalalignment='baseline')
    m.fig.savefig(outpath, bbox_inches='tight')
    #plt.show()
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
    g.plot(sns.scatterplot, sns.histplot, alpha=.2)
    g.fig.suptitle('Relationship between model weights \n'
                   'and ground truth weights for each component',
                   verticalalignment='top')
    # save the plot
    g.fig.tight_layout()
    outpath = Path(outdir) / \
              f'model-weights_versus_ground-truth_all-ds.svg'
    g.fig.savefig(outpath, bbox_inches='tight')


def simulate(n=15,
             percent_nosignal=0.3,
             offset=False,
             space='time-resolved',
             k=3,
             outdir='/tmp',
             weights={},
             raw=None,
             signal=None,
             nsignal=1
             ):
    """
    Simulate data
    :param n: int, number of subjects to simulate
    :param percent_nosignal: How many sensors should not carry signal components
    :param space: Whether or not the data is transformed to its power spectrum
    :param k: number of features for SRM
    :param weights: dict, predefined weights to use for weighting sensors.
     Must be given when raw is not None
    :param raw: list or None, pre-existing raw data to reuse for model fitting
    :param signal: list or None, pre-existing signal to reuse. Must be given
    when raw is not None
    :param nsignal: int, number of signals in the simulated data
    :return:
    """
    if raw and (not signal or not weights):
        raise ValueError("When providing existing raw data to simulate(), you"
                         "must also provide existing signal and weights.")

    if not raw:
        # simulating new data
        if offset:
            # make data without any offset:
            simulated_data = \
                [simulate_raw(make_signal(stype='else',
                                          theta=theta,
                                          nsignal=nsignal),
                              s_without_signal=percent_nosignal,
                              weights=weights.get(i, None))
                    for i, theta in enumerate(rng.uniform(0, 9000, n).astype(int))]
        else:
            simulated_data = [simulate_raw(make_signal(stype='else',
                                                       nsignal=nsignal),
                                           s_without_signal=percent_nosignal,
                                           weights=weights.get(i, None))
                              for i in range(n)]
        raw, weights, signal = list(zip(*simulated_data))
    if space == 'spectral':
        power = transform_to_power(raw)
        model = fit_srm(power, features=k)
    else:
        model = fit_srm(raw, features=k)
        power = None

    plot_srm(model, weights, space=space, outdir=outdir)
    transformed = get_transformations(model, raw, k)
    return raw, weights, signal, model, transformed, power


def plot_simulation(signal, transformed, sub, weights, model):
    nplots = len(transformed) + 1
    fig, ax = plt.subplots(nplots, sharex=True, figsize=(20,15))
    palette = sns.color_palette('husl', nplots)

    ax[0].plot(signal, color=palette[0])
    correlations = {}

    for i in range(len(transformed)):
        ax[i+1].plot(transformed[i][sub], color=palette[i+1])
        # include a correlation in the figure
        corr_modelweight_trueweight = spearmanr(model.w_[sub][:, i],
                                                weights[sub])
        corr_signal_reconstruction = spearmanr(transformed[i][sub],
                                               signal)
        correlations[f'C{i+1}  Model and true weights:'] = \
            round(corr_modelweight_trueweight[0], 2)
        correlations[f'C{i+1}  Signal and reconstruction:'] = \
            round(corr_signal_reconstruction[0], 2)
        ax[i+1].set_title(f"Component {i+1}", verticalalignment='bottom', y=0.9)
        text = \
            r"Correlations: R$_{mw,sw}$=%.2f, R$_{s,r}$=%.2f" % \
            (corr_modelweight_trueweight[0], corr_signal_reconstruction[0])
        ax[i+1].text(0, 0.05, text, horizontalalignment='left',
                     verticalalignment='center', transform=ax[i+1].transAxes)

    for a in ax:
        a.set(ylabel='amplitude')
        a.set_ylim(-5, 5)
    ax[-1].set(xlabel='samples')
    fig.suptitle('Original signal (top) and reconstructed signal',
                   verticalalignment='bottom')
    fig.tight_layout()
    plt.show()
    msg = ""
    for k, v in correlations.items():
        msg += f"{k} {v}\n"
    return correlations


def get_transformations(model, raw, comp):
    """
    Transform raw data into model space.
    :param model:
    :param raw:
    :param comp: number of components in the model
    :return:
    """
    transformations = {}
    for k in range(comp):
        transformations[k] = \
            [np.dot(model.w_[i].T, raw[i])[k] for i in range(len(raw))]
    return transformations



def letsroll(n=15, k=3, percent_nosignal=0.3, outdir='/tmp', nsignal=1):
    """Run the simulations.
    :param n: int, number of subjects to simulate
    :param k: int, number of features for SRM
    :param percent_nosignal: float, between 0 and 1. Percent of sensors without
     signal
     """

    # make data with no offset
    raw, weights, signal, model, transformed, power = \
        simulate(n=n,
                 percent_nosignal=percent_nosignal,
                 offset=False,
                 space='time-resolved',
                 k=k,
                 outdir=outdir,
                 nsignal=nsignal)
    print("Raw data, no offset:")
    QC(raw, weights, signal, model, transformed)
    # now with offset
    raw, weights, signal, model, transformed, power = \
        simulate(n=n,
                 percent_nosignal=percent_nosignal,
                 offset=True,
                 space='time-resolved',
                 k=k,
                 outdir=outdir,
                 nsignal=nsignal)
    print("Raw data, with offset:")
    QC(raw, weights, signal, model, transformed)
    # now with offset and power spectrum transformation
    raw, weights, signal, model, transformed, power = \
        simulate(n=n,
                 percent_nosignal=percent_nosignal,
                 offset=True,
                 space='spectral',
                 k=k,
                 outdir=outdir,
                 nsignal=nsignal)
    print("Spectral data, with offset:")
    QC(raw, weights, signal, model, transformed)

    # use the same data, once time-resolved, once spectral
    raw, weights, signal, model, transformed, power = \
        simulate(n=n,
                 percent_nosignal=percent_nosignal,
                 offset=True,
                 space='time-resolved',
                 k=k,
                 outdir=outdir,
                 nsignal=nsignal)
    print("Same data, time-resolved and with offset:")
    QC(raw, weights, signal, model, transformed)
    # reuse previously simulated data for this run
    raw, weights, signal, model, transformed, power = \
        simulate(n=n,
                 percent_nosignal=percent_nosignal,
                 offset=True,
                 space='spectral',
                 k=k,
                 outdir=outdir,
                 raw=raw,
                 signal=signal,
                 weights=weights,
                 nsignal=nsignal)
    print("Same data, spectral and with offset:")
    QC(raw, weights, signal, model, transformed)

    # use identical weights for each subject. Check whether the QC obtained
    # with a phase-locked signal and the transformation obtained with the same
    # signal with random phase shifts but power spectrum transformations are
    # comparable.
    # make subject weights:
    subject_weights = {i: rng.uniform(0, 1, 306) for i in range(n)}
    raw, weights, signal, model, transformed, power = \
        simulate(n=n,
                 percent_nosignal=percent_nosignal,
                 offset=False,
                 space='time-resolved',
                 k=k,
                 outdir=outdir,
                 weights=subject_weights,
                 nsignal=nsignal)
    print("Raw data, no offset, fixed weights:")
    QC(raw, weights, signal, model, transformed)
    # comparable to
    raw, weights, signal, model, transformed, power = simulate(n=n,
                                                      percent_nosignal=percent_nosignal,
                                                      offset=True,
                                                      space='spectral',
                                                      k=k,
                                                      outdir='/tmp',
                                                      weights=subject_weights,
                                                      nsignal=nsignal)
    print("Spectral data, with offset, fixed weights:")
    QC(raw, weights, signal, model, transformed)



def QC(raw, weights, signal, model, transformed):
    """
    Compute QC plots and metrics as correlations between model results and
    underlying data
    :param raw: array of list, either raw data or raw data converted into power
    :param weights: tuple, contains original weights used in signal generation
    :param signal: tuple, contains original signal
    :param model: SRM model
    :param transformed: dict, for each model component (keys), the resulting
     transformation of raw data into shared space
    :return:
    """
    model_weights_versus_true_weights = defaultdict(list)
    signal_versus_reconstruction = defaultdict(list)
    for i in range(len(raw)):
        correlations = plot_simulation(signal[i][0],
                                       transformed,
                                       i,
                                       weights,
                                       model)
        for k,v in correlations.items():
            if "Model and true weights" in k:
                model_weights_versus_true_weights[k[:3]].append(v)
            elif "Signal and reconstruction" in k:
                signal_versus_reconstruction[k[:3]].append(v)
    msg = ''
    for metric, name in ([model_weights_versus_true_weights, 'avg (min,max) correlation between model weights and true weights '],
                         [signal_versus_reconstruction, 'avg (min,max) correlation between signal and reconstruction ']):
        for comp, v in metric.items():
            msg += f"\n{name+comp}: {np.round(np.mean(v), 2)} ({np.round(np.min(v), 2)}, {np.round(np.max(v), 2)})"
    print(msg)


if __name__ == '__main__':
    import sys
    # TODO parameter
    letsroll()

    # TODO: take subject trials and use the model transformations to obtain
    #  component weights in the time domain for (hopefully) interpretable
    #  SRM components

    # TODO: take response time, not delay. Check whether there is a correlation
    #  between response time and the (hopefully to be found) SRM component of
    #  interest