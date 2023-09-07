import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mne.io import read_raw_fif
from mne import EvokedArray
from scipy.signal import decimate

from pymento_meg.srm.srm import (
get_general_data_structure,
)
from pymento_meg.utils import _construct_path
from pymento_meg.decoding.base import (
    confusion_magnitude,
    confusion_probability,
    confusion_expectedvalue,
    confusion_choice,
    confusion_id,
    decode,
    hijack_scorer_for_inverse_transformations,
    sliding_averager,
    spatiotemporal_slider,
)

# set and check target infos
known_targets = {'probability': {'prefix': 'P',
                                 'tname': 'LoptProb',
                                 'metric': confusion_probability,
                                 'label': ['10%', '20%', '40%', '80%'],
                                 'chance': 0.25,
                                 'ylims': (0.15, 0.45)},
                 'magnitude': {'prefix': 'M',
                               'tname': 'LoptMag',
                               'metric': confusion_magnitude,
                               'label': ['0.5', '1', '2', '4'],
                               'chance': 0.25,
                               'ylims': (0.15, 0.45)},
                 'expectedvalue': {'prefix': 'EV',
                                   'tname': 'ev',
                                   'metric': confusion_expectedvalue,
                                   'label': ['0.2', '0.4', '0.8'],
                                   'chance': 0.33,
                                   'ylims': (0.25, 0.5)},
                 'choice': {'prefix': 'choice',
                            'tname': 'choice',
                            'metric': confusion_choice,
                            'label': ['1', '2'],  # TODO: recode
                            'chance': 0.5,
                            'ylims': (0.3, 0.99),
                            },
                 'identity': {'prefix': 'ID_',
                              'tname': 'Lchar',
                              'metric': confusion_id,
                              'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                        'I'],
                              'chance': 0.11,
                              'ylims': (0, 0.3)
                              }
                 }


def temporal_decoding(sub,
                      target,
                      n_jobs=-1,
                      dec_factor=1,
                      summary_metric='balanced accuracy',
                      datadir='/data/project/brainpeach/memento-sss',
                      bidsdir='/data/project/brainpeach/memento-bids',
                      workdir='/data/project/brainpeach/decoding',
                      dimreduction=None,
                      k=None,
                      trainrange=None,
                      srmsamples=None,
                      ntrials=4,
                      nsamples=100,
                      n_splits=5,
                      responselocked=False,
                      slidingwindow=None,
                      slidingwindowtype=spatiotemporal_slider,
                      ):
    """
    Perform temporal decoding on a memento subject's time series data in sensor
    space.
    :param sub: str; which subject to decode
    :param target: str; which trial feature to decode (magnitude, probability,
    expectedvalue)
    :param n_jobs: int or None; determines amount of parallel jobs. -1 will use
    all available CPUs
    :param dec_factor: int; by which factor to decimate the time series.
    :param summary_metric: str; which metric to evaluate confusion patterns on.
    e.g., 'balanced accuracy', 'f1', 'precision', 'recall'
    :param datadir: str; where is epoched data
    :param bidsdir: str; where is BIDS data
    :param workdir: str; where to save decoding results
    :param dimreduction: None or str; which form of dimensionality reduction to
     use. Can be None, 'pca', 'srm', 'spectralsrm'.
    :param k: None or int; dimensions to reduce to/features to select
    :param trainrange: None or list of int; if not None needs to be a start
    and end range to subselect training data. *Must* be in 1000Hz samples, e.g.
    [0, 700] for first visual stimulation!
    :param ntrials: int; how many trials of the same type to average together
    :param nsamples: int; how many bootstrapping draws during trial averaging
    :param n_splits: int, number of cross-validation folds
    :param responselocked: bool, whether the underlying data is responselocked
    or not (influences parameters in data selection and plotting)
    :param slidingwindow: None or int; over how many samples to create a sliding
     window during decoding
    :param slidingwindowtype: which custom function to use in a sliding window.
    Currently implemented: spatiotemporal_slider and sliding_averager
    :param spectralsrm: bool, whether SRM is trained on spectral data or not
    :return:
    """
    # define the sampling rate. TODO: read this from the data
    sr = 1000
    if target not in known_targets.keys():
        raise NotImplementedError(f"Can't handle target {target} yet."
                                  f" Know targets: {known_targets.keys()}")

    # get data. If its response locked, the timespan needs to be inverted
    # timespan needs to be in seconds, starting from 0
    timespan = [-1.250, 1.250] if responselocked else [0, 4.500]
    fullsample, data = get_general_data_structure(subject=sub,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition='nobrain-brain',
                                                  timespan=timespan)

    X = np.array([decimate(epoch['normalized_data'], dec_factor)
                  for id, epoch in fullsample[sub].items()])

    y = extract_targets(fullsample,
                        sub=sub,
                        target=known_targets[target]['tname'],
                        target_prefix=known_targets[target]['prefix'])
    del fullsample, data
    if dimreduction is None:
        # set output path and be explicit about no trainrange
        fpath = _construct_path([workdir, f'sub-{sub}/'])
        trainrange = None
    else:
        fpath = _construct_path([workdir, f'sub-{sub}/{dimreduction}/'])
        # determine the time range for training data
        trainrange = [int(i / dec_factor) for i in trainrange] \
            if trainrange is not None else None
    # get the decoding performance
    scores = decode(X,
                    y,
                    metric=known_targets[target]['metric'],
                    n_jobs=n_jobs,
                    n_splits=n_splits,
                    dimreduction=dimreduction,
                    k=k,
                    trainrange=trainrange,
                    srmsamples=srmsamples,
                    nsamples=nsamples,
                    ntrials=ntrials,
                    slidingwindow=slidingwindow,
                    slidingwindowtype=slidingwindowtype,
                    )
    topographies = None
    if topographies is not None and dimreduction in ['srm', 'spectralsrm']:
        # we also want to see the topographies of the components (implemented as
        # hijack_scorer_for_inverse_transformations). Because we're using
        # mne's cross_val_multiscore and not sklearn's cross_validate, we can't
        # pass multiple metrics for scoring, and have to repeat the decoding
        # process
        activation_patterns = decode(X,
                              y,
                              metric=hijack_scorer_for_inverse_transformations,
                              n_jobs=n_jobs,
                              n_splits=n_splits,
                              dimreduction=dimreduction,
                              k=k,
                              trainrange=trainrange,
                              srmsamples=srmsamples,
                              nsamples=nsamples,
                              ntrials=ntrials,
                              slidingwindow=slidingwindow,
                              slidingwindowtype=slidingwindowtype,
                              )
        plot_topographies(activation_patterns,
                          sub=sub,
                          target=target,
                          datadir=datadir,
                          fpath=fpath)

    # save the decoding scores for future use
    np.save(Path(fpath) / f'sub-{sub}_decoding-scores_{target}.npy', scores)

    # plot decoding accuracy over all classes
    acrossclasses = np.asarray(
        [np.nanmean(get_metrics(c, metric=summary_metric))
         for score in scores
         for c in np.rollaxis(score, -1, 0)]).reshape(len(scores),
                                                      scores.shape[-1])
    # the x axis (times) gets more and more messier to get right. We need to
    # account for time shifts if we used a sliding windows.
    # Timeoffset is in seconds
    timeoffset = (slidingwindow * dec_factor) / sr if slidingwindow else 0
    # times get centered on 0 if responselocking was done. We subtract the
    # length of the sliding window
    times = np.asarray(np.arange(acrossclasses.shape[-1]) * dec_factor) \
            + timespan[0] * sr + timeoffset * sr

    reflines = [(0, 'response')] if responselocked \
        else ((0, 'onset stimulus'), (700, 'offset stimulus'),
              (2700, 'onset stimulus'), (3400, 'offset stimulus'))
    shading = None if slidingwindow is None else slidingwindow * dec_factor
    plot_decoding_over_all_classes(acrossclasses,
                                   times=times,
                                   label=target, subject=sub,
                                   metric=summary_metric, figdir=fpath,
                                   chance=known_targets[target]['chance'],
                                   ylim=known_targets[target]['ylims'],
                                   reflines=reflines,
                                   slidingwindow=shading,
                                   )

    # plot average confusion matrix over 100ms time slots
    i = 0
    while i < scores.shape[-1]-(100/dec_factor):
        confm = sum_confusion_matrices(scores, slices=(i, int(i+100/dec_factor)))
        fname = Path(fpath) / \
                f'sub-{sub}_conf-matrix_{target}_{i*dec_factor}-{int((i+100/dec_factor)*dec_factor)}ms.svg'
        plot_confusion_matrix(confm,
                              labels=known_targets[target]['label'],
                              fname=fname)
        i += int(100/dec_factor)

def extract_targets(fullsample, sub, target, target_prefix):
    """
    Extract an array of target values from the fullsample data structure. Likely
    targets are 'LoptMag', 'LoptProb', or 'ev' (expected value); or 'choice'

    Example:
    >>> expected_value = extract_targets(fullsample, '004', 'ev', 'EV')
    >>> magnitude = extract_targets(fullsample, '004', 'LoptMag', 'M')
    :param fullsample: dict; contains the epochs data and trial info in a nested
    dict with subjects as keys and epochs as subkeys.
    Output of get_general_data_structure
    :param sub: str; subject identifier
    :param target: str; either key in trial dictionary (e.g., 'LoptMag') or 'ev'
    :param target_prefix: str; string prefix for target values -> M0.5
    :return: array of target values
    """

    if target == 'ev':
        res = \
            np.asarray([f'{target_prefix}' +
                        str(fullsample[sub][d]['LoptMag'] *
                            fullsample[sub][d]['LoptProb'])
                 for d in fullsample[sub]])
    else:
        res = \
            np.asarray([f'{target_prefix}' + str(fullsample[sub][d][f'{target}'])
                        for d in fullsample[sub]])
    return res


def sum_confusion_matrices(confms, slices=None):
    """

    :param confms: array; Confusion matrix. Dims either with or without cv folds
    (CV x (conf_m x conf_n) x times or (conf_m x conf_n) x times)
    :param slice: tuple; from
    :return:
    """
    if slices is None or len(slices) < 2:
        raise ValueError('Please specify slice as a tuple with the start & end '
                         'slice of the time dimension, e.g., slice=(5, 10)')
    sums = np.sum(confms[..., slices[0]:slices[1]], axis=-1)
    if sums.ndim == 3:
        # there are sums for each fold
        sums = np.sum(sums, axis=0)

    return sums


def get_metrics(confm, metric='balanced accuracy'):
    """
    Return recall, precision, f1 score, balanced accuracy or accuracy
    """
    res = dict()
    res['accuracy'] = confm.diagonal()/confm.sum(axis=1)
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    true_pos = confm.diagonal()
    false_pos = np.sum(confm, axis=0) - true_pos
    false_neg = np.sum(confm, axis=1) - true_pos
    res['precision'] = precision = true_pos / (true_pos + false_pos)
    res['recall'] = recall = true_pos / (true_pos + false_neg)
    # https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    res['balanced accuracy'] = 0.5 * (precision + recall)
    res['f1'] = 2 * (precision*recall)/(precision+recall)
    if metric == 'all':
        return res
    return res.get(metric, f'unkown metric "{metric}"')


def plot_decoding_per_class(scores, times):
    pass


def plot_topographies(activation_patterns, sub, target, fpath, datadir):
    """Take computed activation patterns and plot them as a topography."""
    # use real data to create a fake evoked structure
    fname = Path(datadir) / f'sub-001/meg' / f'sub-001_task-memento_proc-sss_meg.fif'
    raw = read_raw_fif(fname)
    # drop all non-meg sensors from the info object
    picks = raw.info['ch_names'][3:309]
    raw.pick_channels(picks)
    splits, sensors, kdims = activation_patterns.shape
    for k in range(kdims):
        # create one plot per k dimensions with results from all splits
        fig, ax = plt.subplots(1, splits)
        for split in splits:
            data = activation_patterns[0, :, k].reshape(-1, 1)
            fake_evoked = EvokedArray(data, raw.info)
            fig = fake_evoked.plot_topomap(times=0,
                                           title=f"Component {k}",
                                           colorbar=False,
                                           size=2
                                           )
        fname = Path(fpath) / f'sub-{sub}_decoding-{target}_topographies_k-{k}.svg'
        fig.savefig(fname)


def plot_confusion_matrix(confm, labels, normalize=True, fname='/tmp/confm.svg'):
    if normalize:
        confm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]
    cm = sns.heatmap(confm, xticklabels=labels, yticklabels=labels,
                     cmap='YlGnBu', vmin=0.25, vmax=0.5, annot=True)
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted')
    cm.figure.savefig(fname)
    cm.figure.clear()


def plot_decoding_over_all_classes(scores,
                                   times,
                                   label,
                                   subject,
                                   metric='accuracy',
                                   chance=0.25,
                                   ylim=None,
                                   figdir='/tmp',
                                   reflines=((0, 'onset stimulus'),
                                             (700, 'offset stimulus'),
                                             (2700, 'onset stimulus'),
                                             (3400, 'offset stimulus')),
                                   slidingwindow=None
                                   ):
    """
    plot the decoded timeseries.
    :param scores: array; or list of arrays; contains scores for each cv fold
    :param times: list; timeseries representing the time course of the data
    :param label: str; label of the target that was decoded (e.g., magnitude)
    :param subject: str; label of the decoded subject
    :param metric: str; metric of the scoring, used to label the y-axis
    :param chance: int; denotes the chance level as a horizontal line
    :param ylim: tuple; allows to set the y-axis range
    :param reflines: nested tuple; to draw vertical markers of trial events
    :param slidingwindow: int or None, length of the sliding window in x
     coordinates to shade
    :return:
    """

    df = pd.DataFrame(scores.T)
    df['time'] = times
    df_melted = pd.melt(df,
                        id_vars=['time'],
                        value_vars=np.arange(0, len(scores)),
                        value_name=metric)
    ax = sns.relplot(x="time",
                     y=metric,
                     kind='line',
                     data=df_melted,
                     height=9,
                     aspect=16/9)
    ax.set(title=f'temporal decoding of {label} (subject {subject})')
    if ylim is not None:
        ax.set(ylim=ylim)
    ax.refline(y=chance, color='red', linestyle='dotted',
               label=f"chance-level: {chance}")
    if reflines is not None:
        for x, l in reflines:
            color = 'black' if l.startswith('offset') else 'green'
            ax.refline(x=x, color=color, label=l)
            if slidingwindow is not None:
                ax.refline(x=x-slidingwindow, color='gray',
                           alpha=0.3, linestyle='solid',
                           label='sliding window')
                # add a shade the size of the sliding window
                ax.ax.fill_between([x, x-slidingwindow], 0, 1, color='gray',
                                   alpha=0.3)
    ax.add_legend()
    fname = f'decoding_{metric.replace(" ","_")}_l2logreg_{subject}_{label}.svg'
    print(f'saving figure to {figdir}/{fname}...')
    ax.fig.savefig(f'{figdir}/{fname}')
    plt.close('all')


def eval_decoding(subject,
                  datadir='/data/project/brainpeach/memento-sss',
                  bidsdir='/data/project/brainpeach/memento-bids',
                  workdir='/data/project/brainpeach/decoding',
                  figdir='/data/project/brainpeach/decoding'):
    """Perform a temporal decoding analysis with a variety of parameters and
    plot the resulting decoding performance measures. Tested variables are:
    dec_factor, k, srmsamples, ntrials, nsamples, slidingwindow,
    slidingwindowtype"""
    # read in the data once
    fullsample, data = get_general_data_structure(subject=subject,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition='nobrain-brain',
                                                  timespan=[-1.25, 1.25])
    results = {}
    # we do not vary parameter with setup with a priory knowledge.
    # downsample to 200hz, 10xsample -> 50ms sliding window
    dec_factor = 5
    slidingwindow = 10
    trainrange = [int(i / dec_factor) for i in [2000, 2500]]

    # get a parameter combination out of the param generator
    for idx, (ntrial, nsample, slidingwindowtype, dimreduction, k, srmsample) in \
            enumerate(parameter_producer()):
        X = np.array([decimate(epoch['normalized_data'], dec_factor)
                      for id, epoch in fullsample[subject].items()])

        y = extract_targets(fullsample,
                            sub=subject,
                            target='choice',
                            target_prefix='choice')
        logging.info(ntrial, nsample, slidingwindowtype, dimreduction, k,
                     srmsample)
        scores = decode(X,
                        y,
                        metric=confusion_choice,
                        n_jobs=-1,
                        n_splits=5,
                        dimreduction=dimreduction,
                        k=k,
                        # train on first visual stim
                        trainrange=trainrange,
                        srmsamples=srmsample,
                        nsamples=nsample,
                        ntrials=ntrial,
                        slidingwindow=slidingwindow,
                        slidingwindowtype=slidingwindowtype,
                        )

        acrossclasses = np.asarray(
            [np.nanmean(get_metrics(c, metric='balanced accuracy'))
             for score in scores
             for c in np.rollaxis(score, -1, 0)]).reshape(
            len(scores), scores.shape[-1]
        )
        # given the fixed dec_factor=5 and slidingwindow=10, the span
        # covers 400ms prior to the response
        areas, peaks = _eval_decoding_perf(acrossclasses, span=[160, 240])
        results[idx] = {'areas': np.median(areas),
                        'peaks': np.median(peaks), 'ntrial': ntrial,
                        'dimreduction': dimreduction,
                        'nsample': nsample, 'k': k, 'srmsample': srmsample,
                        'windowtype': slidingwindowtype.__name__ if
                        slidingwindowtype is not None else 'None'}

    df_results = pd.DataFrame(results).T
    # save the data frame
    fname = Path(workdir) / f'parameter_optimization_sub-{subject}.csv'
    df_results.to_csv(fname)
    # plotting
    _all_plots(figdir, subject, df_results)


def _all_plots(figdir, subject, df_results, aggregate=False):
    fname = _construct_path([Path(figdir), f'sub-{subject}',
                             f'parameter_optimization_sub-{subject}.svg'])
    logging.info(f'generating figure {fname}...')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df_results[~df_results.k.notna()], x='areas',
                    y='peaks', hue='nsample', style='windowtype', size='ntrial',
                    ax=ax)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    title = f'Parameter search for subject {subject}' if not aggregate else \
        f'Parameter averages across subjects'
    fig.suptitle(title)
    plt.xlabel('avg accuracy 500ms prior response')
    plt.ylabel('peak accuracy')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    if not aggregate:
        # set the same x- and y-axis limits in all plots for comparability
        ax.set_ylim(0.55, 0.9)
        ax.set_xlim(0.50, 0.7)
    plt.tight_layout()
    fig.figure.savefig(fname)

    fname = _construct_path([Path(figdir), f'sub-{subject}',
            f'parameter_optimization_srm_sub-{subject}.svg'])
    logging.info(f'generating figure {fname}...')
    fig = sns.relplot(data=df_results[df_results.dimreduction == 'srm'],
                      x='areas', y='peaks', col='windowtype', row='ntrial',
                      hue='nsample', style='srmsample', size='k')
    title = f'SRM parameter search for subject {subject}' if not aggregate \
        else f'Aggregate SRM parameter averages across subjects'
    fig.figure.suptitle(title)
    fig.set_ylabels('peak accuracy')
    fig.set_xlabels('avg accuracy 500ms prior response')
    if not aggregate:
        # set the same x- and y-axis limits in all plots for comparability
        for ax in fig.axes[0]:
            ax.set_ylim(0.55, 0.9)
            ax.set_xlim(0.50, 0.7)
    plt.tight_layout()
    fig.figure.savefig(fname)

    fname = _construct_path([Path(figdir), f'sub-{subject}',
            f'parameter_optimization_spectralsrm_sub-{subject}.svg'])
    logging.info(f'generating figure {fname}...')
    fig = sns.relplot(data=df_results[df_results.dimreduction == 'spectralsrm'],
                      x='areas', y='peaks', col='windowtype', row='ntrial',
                      hue='nsample', style='srmsample', size='k')
    title = f'Spectral SRM parameter search for subject {subject}' if not aggregate \
        else f'Aggregate spectral SRM parameter averages across subjects'
    fig.figure.suptitle(title)
    fig.set_ylabels('peak accuracy')
    fig.set_xlabels('avg accuracy 500ms prior response')
    if not aggregate:
        # set the same x- and y-axis limits in all plots for comparability
        for ax in fig.axes[0]:
            ax.set_ylim(0.55, 0.9)
            ax.set_xlim(0.50, 0.7)
    plt.tight_layout()
    fig.figure.savefig(fname)

    fname = _construct_path([Path(figdir), f'sub-{subject}',
            f'parameter_optimization_pca_sub-{subject}.svg'])
    print(f'generating figure {fname}...')
    fig = sns.relplot(data=df_results[df_results.dimreduction == 'pca'],
                      x='areas', y='peaks', col='windowtype', row='ntrial',
                      hue='nsample', style='srmsample', size='k')
    title = f'PCA parameter search for subject {subject}' if not aggregate \
        else f'Aggregate PCA parameter averages across subjects'
    fig.figure.suptitle(title)
    fig.set_ylabels('peak accuracy')
    fig.set_xlabels('avg accuracy 500ms prior response')
    if not aggregate:
        # set the same x- and y-axis limits in all plots for comparability
        for ax in fig.axes[0]:
            ax.set_ylim(0.55, 0.9)
            ax.set_xlim(0.50, 0.7)
    plt.tight_layout()
    fig.figure.savefig(fname)

def parameter_producer():
    # for now, only loop over parameter spaces of interest.
    # potentially refine hypothesis-driven later
    for dimreduction in ['pca', None, 'srm', 'spectralsrm']:
        for ntrial in [1, 5, 10]:
            for nsample in ['min', 'max', 500]:
                for slidingwindowtype in [sliding_averager, spatiotemporal_slider, None]:
                    if dimreduction is None:
                        yield ntrial, nsample, slidingwindowtype, \
                              dimreduction, None, None
                    else:
                        for k in [2, 5, 10, 25, 80]:
                            if dimreduction in ['srm', 'spectralsrm']:
                                for srmsample in [10, 20, 50]:
                                        yield ntrial, nsample, \
                                              slidingwindowtype, dimreduction, \
                                              k, srmsample
                            elif dimreduction == 'pca':
                                yield ntrial, nsample, slidingwindowtype, \
                                      dimreduction, k, None


def _eval_decoding_perf(accuracies, span):
    """For a given span of accuracies, compute the area under the curve
     (average accuracy) and peak accuracy"""
    # calulcate area under the curve, normalized by the range of the integral
    #https://www.intmath.com/applications-integration/9-average-value-function.php
    areas = [
        np.trapz(acc[span[0]:span[1]]) / (span[1] - span[0] - 1)
        for acc in accuracies
    ]
    peaks = [np.max(acc[span[0]:span[1]]) for acc in accuracies]
    return areas, peaks


def mean_str(col):
    from pandas.api.types import is_numeric_dtype
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique()[0] if col.nunique() == 1 else np.NaN


def aggregate_evals(
        figdir='/data/project/brainpeach/decoding',
        subject='all',
        fpath='/data/project/brainpeach/decoding/parameter_optimization_sub-*.csv',
        ):
    """Read in all decoding parameter optimization results and plot the average
    results across subjects for each parameter combination. """
    if not (Path(figdir) /f'sub-{subject}').exists():
        from os import makedirs
        import logging
        logging.info(f'Creating {figdir}/sub-{subject}...')
        makedirs(Path(figdir)/f'sub-{subject}')
    from glob import glob
    dfs = []
    for file in glob(fpath):
        df = pd.read_csv(file)
        dfs.append(df)
    df_results = pd.concat(dfs)
    means = df_results.groupby(df_results.index).agg(mean_str)
    _all_plots(figdir=figdir, subject=subject, df_results=means, aggregate=True)


def aggregate_decoding(
        figdir='/data/project/brainpeach/decoding',
        mode='subject'
):
    """
    Create aggregate plots across decoding results. As two mode types: 'subject'
    (aggregating data for any given dimreduction and target over all subjects)
    or 'dimreduction' (aggregating data for any given subject and target over
    all dimreduction methods).
    :param figdir: str, directory where to save figures in
    :param mode: str, must be one of 'subject' or 'dimreduction'. Used to
    determine the mode of plotting/aggregating.
    :return:
    """
    # hardcode the parameters used in decoding
    slidingwindow = 10
    dec_factor = 5
    sr = 1000

    if mode == 'subject':
        # aggregate over subjects
        hue = y = 'subject'
        for target, ylim, chance in \
            [['choice', (0.3, 0.99), 0.5], ['magnitude', (0.2, 0.35), 0.25],
             ['probability',  (0.2, 0.35), 0.25],
             ['expectedvalue', (0.3, 0.4), 0.33],
             #['identity', (0, 0.3), 0.11]
             ]:
            for dimreduction in [None, 'srm', 'pca', 'spectralsrm']:
                # hardcode the parameters used in decoding
                slidingwindow = 10
                dec_factor = 5
                sr = 1000
                timespan = [-1.25, 1.25] if target == 'choice' else [0, 4500]
                dfs = []
                for sub in np.arange(1, 23):
                    subject = f'00{sub}' if sub < 10 else f'0{sub}'
                    fpath = f'sub-{subject}/sub-{subject}_decoding-scores_{target}.npy' \
                        if dimreduction is None else \
                        f'sub-{subject}/{dimreduction}/sub-{subject}_decoding-scores_{target}.npy'

                    scores = np.load(str(Path(figdir) / fpath))
                    acrossclasses = np.asarray(
                        [np.nanmean(get_metrics(c, metric='balanced accuracy'))
                         for score in scores
                         for c in np.rollaxis(score, -1, 0)]).reshape(
                        len(scores), scores.shape[-1]
                    )
                    # the x axis (times) gets more and more messier to get right. We need to
                    # account for time shifts if we used a sliding windows.
                    # Timeoffset is in seconds
                    timeoffset = (slidingwindow * dec_factor) / sr if slidingwindow else 0
                    # times get centered on 0 if responselocking was done. We subtract the
                    # length of the sliding window
                    times = np.asarray(np.arange(acrossclasses.shape[-1]) * dec_factor) \
                            + timespan[0] * sr + timeoffset * sr
                    df = pd.DataFrame(acrossclasses.T)
                    df['subject'] = np.repeat(subject, acrossclasses.shape[-1])
                    df['time'] = times
                    dfs.append(df)
                ax = _plot_aggregated(dfs, y, hue, target, dimreduction,
                                      slidingwindow, dec_factor, chance, ylim,
                                      )
                fname = f'decoding_balanced-accuracy_l2logreg_{target}_dimreduction-{dimreduction}.svg'
                print(f'saving figure to {figdir}/{fname}...')
                ax.fig.savefig(f'{figdir}/{fname}')
                plt.close('all')
                # repeat, but average
                ax = _plot_aggregated(dfs, y, hue, target, dimreduction,
                                      slidingwindow, dec_factor, chance, ylim,
                                      average=True)
                fname = f'averaged-decoding_balanced-accuracy_l2logreg_{target}_dimreduction-{dimreduction}.svg'
                print(f'saving figure to {figdir}/{fname}...')
                ax.fig.savefig(f'{figdir}/{fname}')
                plt.close('all')

    elif mode == 'dimreduction':
        # aggregate over dimreductions
        hue = y = 'dimreduction'
        for sub in np.arange(1, 23):
            subject = f'00{sub}' if sub < 10 else f'0{sub}'
            for target, ylim, chance in \
                    [['choice', (0.3, 0.99), 0.5],
                     ['magnitude', (0.2, 0.35), 0.25],
                     ['probability', (0.2, 0.35), 0.25],
                     ['expectedvalue', (0.3, 0.4), 0.33],
                     #['identity', (0, 0.3), 0.11]
                      ]:
                timespan = [-1.25, 1.25] if target == 'choice' else [0, 4500]
                dfs = []
                for dimreduction in [None, 'srm', 'pca', 'spectralsrm']:
                    fpath = f'sub-{subject}/sub-{subject}_decoding-scores_{target}.npy' \
                        if dimreduction is None else \
                        f'sub-{subject}/{dimreduction}/sub-{subject}_decoding-scores_{target}.npy'

                    scores = np.load(str(Path(figdir) / fpath))
                    acrossclasses = np.asarray(
                        [np.nanmean(get_metrics(c, metric='balanced accuracy'))
                         for score in scores
                         for c in np.rollaxis(score, -1, 0)]).reshape(
                        len(scores), scores.shape[-1]
                    )
                    # the x axis (times) gets more and more messier to get right. We need to
                    # account for time shifts if we used a sliding windows.
                    # Timeoffset is in seconds
                    timeoffset = (slidingwindow * dec_factor) / sr if slidingwindow else 0
                    # times get centered on 0 if responselocking was done. We subtract the
                    # length of the sliding window
                    times = np.asarray(
                        np.arange(acrossclasses.shape[-1]) * dec_factor) \
                            + timespan[0] * sr + timeoffset * sr
                    df = pd.DataFrame(acrossclasses.T)
                    df['dimreduction'] = np.repeat(dimreduction,
                                                   acrossclasses.shape[-1])
                    df['time'] = times
                    dfs.append(df)

                ax = _plot_aggregated(dfs, y, hue, target,
                                      dimreduction='all methods',
                                      slidingwindow=slidingwindow,
                                      dec_factor=dec_factor,
                                      chance=chance,
                                      ylim=ylim,
                                      )
                fname = f'sub-{subject}_decoding_balanced-accuracy_l2logreg_{target}_dimreduction-all.svg'
                print(f'saving figure to {figdir}/sub-{subject}/{fname}...')
                ax.fig.savefig(f'{figdir}/sub-{subject}/{fname}')
                # repeat, but average
                ax = _plot_aggregated(dfs, y, hue, target,
                                      dimreduction='average over methods',
                                      slidingwindow=slidingwindow,
                                      dec_factor=dec_factor,
                                      chance=chance, ylim=ylim,
                                      average=True)
                fname = f'sub-{subject}_averaged-decoding_balanced-accuracy_l2logreg_{target}.svg'
                print(f'saving figure to {figdir}/sub-{subject}/{fname}...')
                ax.fig.savefig(f'{figdir}/sub-{subject}/{fname}')
                plt.close('all')


def _plot_aggregated(dfs,
                     y,
                     hue,
                     target,
                     dimreduction,
                     slidingwindow,
                     dec_factor,
                     chance,
                     ylim=None,
                     average=False
                     ):
    # combine the data and plot
    allscores = pd.concat(dfs)
    fulldf = pd.melt(allscores,
                     id_vars=['time', y],
                     value_vars=np.arange(0, 5),
                     value_name='accuracy')
    offset = shading = slidingwindow * dec_factor
    reflines =[(0 + offset, 'response')] if target == 'choice' else \
        ((0 + offset, 'onset stimulus'), (700 + offset, 'offset stimulus'),
         (2700 + offset, 'onset stimulus'), (3400 + offset, 'offset stimulus'))
    plt_args = dict(x="time", y='accuracy', kind='line', data=fulldf, height=9,
                    aspect=16/9, alpha=0.3)
    if not average:
        plt_args['hue'] = hue
    ax = sns.relplot(**plt_args)
    if ylim is not None:
        ax.set(ylim=ylim)
    dimred = 'without dimensionality reduction' if dimreduction == None else dimreduction
    ax.set(title=f'temporal decoding of {target} ({dimred})')
    if reflines is not None:
        for x, l in reflines:
            color = 'black' if l.startswith('offset') else 'green'
            ax.refline(x=x, color=color, label=l)
        if shading is not None:
            ref = reflines[0][0]
            ax.refline(x=ref-shading, color='gray',
                       alpha=0.3, linestyle='solid',
                       label='sliding window')
                # add a shade the size of the sliding window
            ax.ax.fill_between([ref, ref-shading], 0, 1, color='gray',
                               alpha=0.3)
    ax.refline(y=chance, color='red', linestyle='dotted',
               label=f"chance-level: {chance}")
    handles, labels = ax.ax.get_legend_handles_labels()
    if target == 'choice':
        # subset legend to avoid duplicatoin
        plt.legend(handles[-3:], labels[-3:])
    else:
        plt.legend(handles[-4:], labels[-4:])
    return ax


def regress_behavior_on_reinstatement(bidsdir, decoding_dir):
    from pymento_meg.proc.behavior import logreg
    from statsmodels.formula.api import ols
    # get the beta weights for stimulus properties driving choice behavior
    coefs = logreg(bidsdir)
    subjects = ['001', '002', '003', '004', '005', '006', '007', '008', '009',
               '010', '011', '012', '013', '014', '015', '016', '017', '018',
               '019', '020', '021', '022']
    vars = ['LoptProb', 'LoptMag', 'l_ev'],
    res = pd.DataFrame()
    for target in ['expectedvalue', 'magnitude', 'probability', 'identity']:
        # perform one regression per decoding target that allows a statement
        # about reinstatement
        designmatrix = pd.DataFrame()
        scoring = {}
        # generate the static part of the design matrix:
        for sub in subjects:
            behav = {label: beta for label, beta in zip(
                ['LoptProb', 'LoptMag', 'l_ev', 'RoptProb', 'RoptMag', 'r_ev'],
                coefs[sub]['pure_coefs']
            )}
            designmatrix = \
                designmatrix.append({'sub': sub, **behav}, ignore_index=True)

            # read in the data, and cache it
            fname = Path(decoding_dir) / f'sub-{sub}' / \
                    f'sub-{sub}_decoding-scores_{target}.npy'
            scores = np.load(fname)
            acrossclasses = np.asarray(
                [np.nanmean(get_metrics(c, metric='balanced accuracy'))
                 for score in scores
                 for c in np.rollaxis(score, -1, 0)]).reshape(
                len(scores), scores.shape[-1]
            )
            scoring[sub] = acrossclasses.mean(axis=0)
        # a counter for samples
        s = 0
        while s < len(acrossclasses[-1]) - 1:
            designmatrix['reinstatement'] = [scoring[subject][s] for subject in subjects]
            print(f'running on sample {s}')
            model = \
                ols('reinstatement ~ LoptProb + LoptMag + l_ev',
                    designmatrix).fit()
            betas = {label: beta for label, beta in zip(
                ['intercept', 'LoptProb', 'LoptMag', 'l_ev'],
                model._results.params
            )}
            ts = {label: beta for label, beta in zip(
                ['t_intercept', 't_LoptProb', 't_LoptMag', 't_l_ev'],
                model._results.tvalues
            )}
            ps = {label: beta for label, beta in zip(
                ['p_intercept', 'p_LoptProb', 'p_LoptMag', 'p_l_ev'],
                model._results.pvalues
            )}
            res = res.append({**betas, **ts, **ps, 'target': target}, ignore_index=True)
            s += 1

    for target in ['expectedvalue', 'magnitude', 'probability', 'identity']:
        df = res[res.target == target]
        times = np.arange(0, 889) * 5
        df['times'] = times
        t_vals = pd.melt(df, id_vars=['times'],
                         value_vars=['t_LoptProb', 't_LoptMag', 't_l_ev'],
                         value_name='t_value')
        plt_args = dict(x="times", y='t_value', hue='variable', kind='line',
                        data=t_vals, height=9, aspect=16/9, alpha=0.8)
        ax = sns.relplot(**plt_args)
        ax.set(ylim=[-4, 4])
        ax.refline(y=1.96, color='red', linestyle='dotted',
               label=f"statistical significance")
        ax.refline(y=-1.96, color='red', linestyle='dotted',
               label=f"statistical significance")
        reflines = ((0, 'onset stimulus'), (700, 'offset stimulus'),
                    (2700, 'onset stimulus'), (3400, 'offset stimulus'))
        for x, l in reflines:
            color = 'black' if l.startswith('offset') else 'green'
            ax.refline(x=x, color=color, label=l)
        ax.set(title=f"Relationship of neural representation of {target},"
                     f" and influence of stimulus parameter on choice")
        ax.savefig(f'/home/adina/scratch/test-{target}.svg')