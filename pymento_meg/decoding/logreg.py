from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    decode,
    sliding_averager,
    spatiotemporal_slider,
)


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
                      srmtrainrange=None,
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
    :param dimreduction: None or sklearn transformer
    :param k: None or int; dimensions to reduce to/features to select
    :param srmtrainrange: None or list of int; if not None needs to be a start
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
    :return:
    """
    # define the sampling rate. TODO: read this from the data
    sr = 1000
    # set and check target infos
    known_targets = {'probability': {'prefix': 'P',
                                     'tname':'LoptProb',
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
                                'label': ['1', '2'], # TODO: recode
                                'chance': 0.5,
                                'ylims': (0.3, 0.99),
                                }
                     }
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
    if dimreduction is not None:
        fpath =_construct_path([workdir, f'sub-{sub}/{dimreduction}/'])
        if dimreduction == 'srm':
            # determine the time range for training data
            trainrange = [int(i/dec_factor) for i in srmtrainrange] \
                if srmtrainrange is not None else None
        else:
            trainrange = None
    else:
        fpath =_construct_path([workdir, f'sub-{sub}/'])
        trainrange = None

    scores = decode(X,
                    y,
                    metric=known_targets[target]['metric'],
                    n_jobs=n_jobs,
                    n_splits=n_splits,
                    dimreduction=dimreduction,
                    k=k,
                    # train on first visual stim
                    srmtrainrange=trainrange,
                    srmsamples=srmsamples,
                    nsamples=nsamples,
                    ntrials=ntrials,
                    slidingwindow=slidingwindow,
                    slidingwindowtype=slidingwindowtype,
                    )

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
                f'sub-{sub}_conf-matrix_{target}_{i*dec_factor}-{int((i+100/dec_factor)*dec_factor)}ms.png'
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

def plot_confusion_matrix(confm, labels, normalize=True, fname='/tmp/confm.png'):
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
    fname = f'decoding_{metric.replace(" ","_")}_l2logreg_{subject}_{label}.png'
    print(f'saving figure to {figdir}/{fname}...')
    ax.fig.savefig(f'{figdir}/{fname}')
    plt.close('all')


def eval_decoding(subject,
                  datadir,
                  bidsdir,
                  figdir='/tmp'):
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
    # we do not vary parameter with setup with a priory knowledge
    dec_factor = 5
    slidingwindow = 10
    # parameter spaces of interest
    ntrials = [1, 3, 5]
    nsamples = ['min', 'max', 300]
    slidingwindowtypes = [sliding_averager, spatiotemporal_slider]
    ks = [10, 25, 50, 80]
    srmsamples = [10, 20, 50]
    results = {}
    results_srm = {}
    run = 0
    srmrun = 0
    srmtrainrange = [int(i / dec_factor) for i in [2000, 2500]]
    # and now loop:
    for ntrial in ntrials:
        for nsample in nsamples:
            for slidingwindowtype in slidingwindowtypes:
                run += 1
                X = np.array([decimate(epoch['normalized_data'], dec_factor)
                              for id, epoch in fullsample[subject].items()])

                y = extract_targets(fullsample,
                                    sub=subject,
                                    target='choice',
                                    target_prefix='choice')

                scores = decode(X,
                                y,
                                metric=confusion_choice,
                                n_jobs=-1,
                                n_splits=5,
                                nsamples=nsample,
                                ntrials=ntrial,
                                slidingwindow=slidingwindow,
                                slidingwindowtype=slidingwindowtype
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
                results[run] = {'areas': np.median(areas),
                                'peaks': np.median(peaks), 'ntrial': ntrial,
                                'nsample': nsample,
                                'windowtype': slidingwindowtype.__name__}
                for k in ks:
                    for srmsample in srmsamples:
                        srmrun += 1
                        scores = decode(X,
                                        y,
                                        metric=confusion_choice,
                                        n_jobs=-1,
                                        n_splits=5,
                                        dimreduction='srm',
                                        k=k,
                                        # train on first visual stim
                                        srmtrainrange=srmtrainrange,
                                        srmsamples=srmsample,
                                        nsamples=nsample,
                                        ntrials=ntrial,
                                        slidingwindow=slidingwindow,
                                        slidingwindowtype=slidingwindowtype,
                                        )
                        acrossclasses = np.asarray(
                            [np.nanmean(
                                get_metrics(c, metric='balanced accuracy'))
                             for score in scores
                             for c in np.rollaxis(score, -1, 0)]).reshape(
                            len(scores), scores.shape[-1]
                        )
                        areas, peaks = _eval_decoding_perf(acrossclasses,
                                                           span=[180, 240])
                        results_srm[srmrun] = {'areas': np.median(areas),
                                               'peaks': np.median(peaks),
                                               'ntrial': ntrial,
                                               'nsample': nsample,
                                               'windowtype': slidingwindowtype.__name__,
                                               'k': k, 'srmsample': srmsample}

    df_results = pd.DataFrame(results).T
    fname = Path(figdir) / f'parameter_optimization_sub-{subject}.png'
    print(f'generating figure {fname}...')
    fig, ax = plt.subplots(figsize = (9, 5))
    sns.scatterplot(data=df_results, x='areas', y='peaks', hue='nsample',
                    style='windowtype', size='ntrial', ax=ax)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.suptitle(f'Parameter search for subject {subject}')
    plt.tight_layout()
    fig.figure.savefig(fname)

    df_results_srm = pd.DataFrame(results_srm).T
    fname = Path(figdir) / f'parameter_optimization_srm_sub-{subject}.png'
    print(f'generating figure {fname}...')
    fig = sns.relplot(data=df_results_srm, x='areas', y='peaks', col='windowtype',
                      row='ntrial', hue='nsample', style='srmsample', size='k')
    fig.figure.suptitle(f'SRM parameter search for subject {subject}')
    plt.tight_layout()
    fig.figure.savefig(fname)



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