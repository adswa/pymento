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
    decode
)


def temporal_decoding(sub,
                      target,
                      average_trials=None,
                      n_jobs=-1,
                      dec_factor=1,
                      summary_metric='balanced accuracy',
                      datadir='/data/project/brainpeach/memento-sss',
                      bidsdir='/data/project/brainpeach/memento-bids',
                      workdir='/data/project/brainpeach/decoding'):
    """
    Perform temporal decoding on a memento subject's time series data in sensor
    space.
    :param sub: str; which subject to decode
    :param target: str; which trial feature to decode (magnitude, probability,
    expectedvalue)
    :param average_trials: int or None; average this many trials of a type for
    higher signal to noise ratio
    :param n_jobs: int or None; determines amount of parallel jobs. -1 will use
    all available CPUs
    :param dec_factor: int; by which factor to decimate the time series.
    :param summary_metric: str; which metric to evaluate confusion patterns on.
    e.g., 'balanced accuracy', 'f1', 'precision', 'recall'
    :param datadir: str; where is epoched data
    :param bidsdir: str; where is BIDS data
    :param workdir: str; where to save decoding results
    :return:
    """

    # set and check target infos
    known_targets = {'probability': {'prefix': 'P',
                                     'tname':'LoptProb',
                                     'metric': confusion_probability,
                                     'label': ['10%', '20%', '40%', '80%'],
                                     'chance': 0.25,
                                     'ylims': (0.1, 0.6)},
                     'magnitude': {'prefix': 'M',
                                   'tname': 'LoptMag',
                                   'metric': confusion_magnitude,
                                   'label': ['0.5', '1', '2', '4'],
                                   'chance': 0.25,
                                   'ylims': (0.1, 0.6)},
                     'expectedvalue': {'prefix': 'EV',
                                       'tname': 'ev',
                                       'metric': confusion_expectedvalue,
                                       'label': ['0.2', '0.4', '0.8'],
                                       'chance': 0.33,
                                       'ylims': (0.2, 0.7)}
                     }
    if target not in known_targets.keys():
        raise NotImplementedError(f"Can't handle target {target} yet."
                                  f" Know targets: {known_targets.keys()}")

    # get data
    fullsample, data = get_general_data_structure(subject=sub,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition='nobrain-brain',
                                                  timespan=[0, 4500])

    X = np.array([decimate(epoch['normalized_data'], dec_factor)
                  for id, epoch in fullsample[sub].items()])

    y = extract_targets(fullsample,
                        sub=sub,
                        target=known_targets[target]['tname'],
                        target_prefix=known_targets[target]['prefix'])

    if average_trials is not None:
        X, y = average_n_trials(X, y, foldsize=average_trials)

    scores = decode(X, y, metric=known_targets[target]['metric'],
                    n_jobs=n_jobs, n_splits=5)

    # save the decoding scores for future use
    fpath =_construct_path([workdir, f'sub-{sub}/'])
    np.save(Path(fpath) / f'sub-{sub}_decoding-scores_{target}.npy', scores)

    # plot decoding accuracy over all classes
    acrossclasses = np.asarray(
        [np.nanmean(get_metrics(c, metric=summary_metric))
         for score in scores
         for c in np.rollaxis(score, -1, 0)]).reshape(len(scores),
                                                      scores.shape[-1])

    plot_decoding_over_all_classes(acrossclasses,
                                   times=np.asarray(
                                       np.arange(0, 4501/dec_factor)
                                       * dec_factor),
                                   label=target, subject=sub,
                                   metric=summary_metric, figdir=fpath,
                                   chance=known_targets[target]['chance'],
                                   ylim=known_targets[target]['ylims'])

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
    targets are 'LoptMag', 'LoptProb', or 'ev' (expected value)

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


def average_n_trials(X, y, foldsize=4):
    """
    Increase the signal to noise ratio by averaging n=foldsize trials per target
    together. This creates averages pseudotrials.

    :param X:
    :param y:
    :return:
    """
    X_ = []
    y_ = []
    for unique_target in np.unique(y):
        # average trials in batches of the specified foldsize per unique target
        trials = X[np.where(y == unique_target)]
        # average the trials
        avg = _average_n_trials(trials, foldsize)
        X_.append(avg)
        # create the appropriate amount of targets
        y_.append(np.repeat(unique_target, len(avg)))
    X_ = np.concatenate(X_)
    y_ = np.concatenate(y_)
    assert len(X_) == len(y_)
    return X_, y_


def _average_n_trials(trials, foldsize=4):
    """
    Average a specified amount of trials together to reduce noise. A set of
    32 trials would be reduced to 8 averaged trials when averaging with a
    foldsize of 4. Make sure that the supplied trials all have the same target!

    :param trials: array-like; list of trial arrays
    :param foldsize: int; number of trials to average over
    """
    res = []
    i = 1
    stop = False

    while not stop:
        # prevent an index error. Beware that this might lead to trials at the
        # end can be averaged in two different average trials, the last and the
        # second to last.
        if i + foldsize > len(trials):
            i = len(trials[:-4])
            stop = True
        mean = np.mean(trials[i:i+foldsize], axis=0)
        res.append(mean)
        i += 4

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
                     cmap='YlGnBu', vmin=0.2, vmax=0.6)
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
                     errorbar='se',
                     data=df_melted,
                     height=9,
                     aspect=16/9)
    ax.set(title=f'temporal decoding of {label} (subject {subject})')
    if ylim is not None:
        ax.set(ylim=ylim)
    ax.refline(y=chance, color='red', linestyle='dotted')
    for x, l in reflines:
        color = 'black' if l.startswith('offset') else 'green'
        ax.refline(x=x, color=color, label=l)
    ax.add_legend()
    fname = f'decoding_{metric.replace(" ","_")}_l2logreg_{subject}_{label}.png'
    print(f'saving figure to {figdir}/{fname}...')
    ax.fig.savefig(f'{figdir}/{fname}')
    plt.close('all')


