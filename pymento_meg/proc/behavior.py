"""
This script exists to describe, model, and analyze behavioral data.
Its mostly a placeholder for now.
Input by more experienced cognitive scientists is probably useful before I start
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from mne.decoding import cross_val_multiscore
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from pymento_meg.orig.behavior import read_bids_logfile


def bigdf(bidsdir):
    """
    aggregate all log files into one big dataframe
    :param bidsdir:
    :return:
    """
    # need a list of sub ids, can't think of something less convoluted atm
    subs = sorted([sub[-3:] for sub in glob(bidsdir + '/' + 'sub-*')])
    dfs = []
    for subject in subs:
        df = read_bids_logfile(subject, bidsdir)
        # add a subject identifier
        df['subject'] = subject
        dfs.append(df)
    # merge the dfs. Some subjects have more column keys than others, join=outer
    # fills them with nans where they don't exist
    return pd.concat(dfs, axis=0, join='outer')


def global_stats(bidsdir):
    """
    Compute statistics from the complete set of log files, over data from all
    subjects
    :param bidsdir:
    :return:
    """
    results = {}
    df = bigdf(bidsdir)
    results['mean'] = np.nanmean(df['RT'])
    results['median'] = np.nanmedian(df['RT'])
    results['std'] = np.nanstd(df['RT'])
    # TODO: continue here with more stats


def stats_per_subject(subject, bidsdir, results=None):
    """
    Compute summary statistics for a subject
    :param subject:
    :param bidsdir:
    :return:
    """
    df = read_bids_logfile(subject, bidsdir)
    if results is None:
        results = {}
    # median reaction time over all trials
    results[subject] = {'median RT global': np.nanmedian(df['RT'])}
    results[subject] = {'mean RT global': np.nanmean(df['RT'])}
    results[subject] = {'Std RT global': np.nanstd(df['RT'])}
    # no-brainer trials
    right = df['RT'][(df.RoptMag > df.LoptMag) &
                     (df.RoptProb > df.LoptProb)].values
    left = df['RT'][(df.LoptMag > df.RoptMag) &
                    (df.LoptProb > df.RoptProb)].values
    nobrainer = np.append(right, left)
    results[subject] = {'median RT nobrainer': np.nanmedian(nobrainer)}
    results[subject] = {'mean RT nobrainer': np.nanmean(nobrainer)}
    results[subject] = {'Std RT nobrainer': np.nanstd(nobrainer)}
    # TODO: continue here with more stats


def logreg(bidsdir,
           figdir='/tmp',
           n_splits=10,
           subject=None):
    """
    Run a logistic regression of stimulus characteristics (left and right
    magnitude, probability, and expected value) on stimulus choice (left or
    right). The analysis is per subject, and reports model quality as accuracy
    and normalized parameter weights to gauge their relative importance to
    identify different choice strategies (e.g., "only rely on probability")
    :param bidsdir: str; where do the behavioral log files reside?
    :param figdir: str; where shall figures be saved?
    :param n_splits: int; number of folds in the cross validation
    :param subject: list of str; for which subjects to compute the analysis
    :return:
    """
    # by default, analyze all subjects
    if subject is None:
        subject = ['001', '002', '003', '004', '005', '006', '007', '008', '009',
                   '010', '011', '012', '013', '014', '015', '016', '017', '018',
                   '019', '020', '021', '022']
    if type(subject) != list:
        subject = [subject]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(fit_intercept=True))
    cv = StratifiedKFold(n_splits=n_splits)
    # the order in which to report parameters and their names/labels in the plot
    stim_chars = ['LoptProb', 'LoptMag', 'l_ev', 'RoptProb', 'RoptMag', 'r_ev']
    labels = ['Prob(L)', 'Mag(L)', 'EV(L)', 'Prob(R)', 'Mag(R)', 'EV(R)']
    coefs = {}
    for sub in subject:
        df = read_bids_logfile(sub, bidsdir)
        # add expected value variables to the data frame, calculated from
        # demeaned left and right stimulation
        df['l_ev'] = (df.LoptProb - df.LoptProb.mean()) * \
                     (df.LoptMag - df.LoptMag.mean())
        df['r_ev'] = (df.RoptProb - df.RoptProb.mean()) * \
                     (df.RoptMag - df.RoptMag.mean())
        # Get data & targets from trials in which a choice was made
        # (0 == no choice)
        X = df[stim_chars][df['choice'] != 0].values
        y = df['choice'][df['choice'] != 0].values
        # choice is coded 1 or 2 for left and right; recode it to 1, 0
        y[np.where((y == 1))] = 0
        y[np.where((y == 2))] = 1
        # perform the classification. Use default scoring metric accuracy, but
        # also return the parameter weights
        scores = cross_val_multiscore(clf, X, y, cv=cv, scoring=getmecoefs)
        # average accuracies across folds
        avg_acc = np.mean([scores[i][0] for i in range(len(scores))])
        # extract & normalize coefficients from each fold to create boxplots
        # from them. The coefficient sum is set to 1.
        # First, get normalized coefficients from each fold (for boxplot)
        coefs_for_boxplot = np.split(
            np.asarray(
                [(np.abs(scores[i][1:][0][0]) /
                  np.sum(np.abs(scores[i][1:][0][0])))[k]
                 for k in range(X.shape[1]) for i in range(len(scores))]
                ),
            X.shape[1])
        # next, calculate average across folds for each parameter, as a label
        avg_coefs = np.mean(
            [np.abs(scores[i][1:][0][0]) / np.sum(np.abs(scores[i][1:][0][0]))
             for i in range(len(scores))],
            axis=0
        )
        # create the boxplots
        print_coefs(data=coefs_for_boxplot,
                    means=avg_coefs,
                    names=labels,
                    sub=sub,
                    acc=avg_acc,
                    figdir=figdir)
        # keep all the coefficients for later
        coefs[sub] = [avg_acc, avg_coefs]
    return coefs


def getmecoefs(est, X, y_true, **kwargs):
    """
    custom scorer to retrieve accuracies and coefficients from log. regression
    """
    y_pred = est.predict(X)
    acc = metrics.accuracy_score(y_true, y_pred)
    coefs = est.steps[1][1].coef_
    return acc, coefs


def print_coefs(data, means, names, sub, acc, figdir='/tmp'):
    """
    Visualize parameter importance with boxplots
    :param data: list of arrays; each array has coefficients for parameters from
     several folds of a cv
    :param means: array; contains the average coefficient for each parameter.
    Used to annotate the boxplots
    :param names: list of str; used to label the x-axis
    :param sub: str; used in the plot title
    :param acc: float; accuracy of the model. Used in the plot title
    :param figdir: str; where shall figures be saved
    :return:
    """
    fig, ax = plt.subplots()
    bplots = ax.boxplot(data, labels=names)
    ax.set_ylim(0, 0.5)
    # add mean value annotation to each boxplot center
    for idx, box in enumerate(bplots['medians']):
        x, y = box.get_xydata()[1]
        plt.text(x, y, '%.2f' % means[idx], verticalalignment='center')
    plt.ylabel('normalized coefficients')
    plt.title(f'Log. reg of stimulus params on choice, sub-{sub}.'
              f' Acc: {acc:.2f}')
    fname = Path(figdir) / f'logreg_stimulus-params-choice_subject-{sub}.png'
    logging.info(f'Saving a boxplot of parameter importance into {fname}.')
    plt.savefig(fname)
    plt.close('all')
