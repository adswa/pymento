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
    plot_decoding_over_all_classes(acrossclasses,
                                   times=times,
                                   label=target, subject=sub,
                                   metric=summary_metric, figdir=fpath,
                                   chance=known_targets[target]['chance'],
                                   ylim=known_targets[target]['ylims'],
                                   reflines=reflines,
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
                                             (3400, 'offset stimulus'))
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
    ax.add_legend()
    fname = f'decoding_{metric.replace(" ","_")}_l2logreg_{subject}_{label}.png'
    print(f'saving figure to {figdir}/{fname}...')
    ax.fig.savefig(f'{figdir}/{fname}')
    plt.close('all')
