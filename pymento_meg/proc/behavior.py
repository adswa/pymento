"""
This script exists to describe, model, and analyze behavioral data.
Its mostly a placeholder for now.
Input by more experienced cognitive scientists is probably useful before I start
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
from scipy.stats import spearmanr
from mne.decoding import cross_val_multiscore
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from pymento_meg.orig.behavior import read_bids_logfile
from pymento_meg.decoding.logreg import get_metrics
from pymento_meg.utils import _construct_path

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


def stats_per_subject(subject, bidsdir=None, df=None):
    """
    Compute summary statistics for a subject
    :param subject:
    :param bidsdir:
    :return:
    """
    if df is None:
        df = read_bids_logfile(subject, bidsdir)
    results = {}
    # median reaction time over all trials
    results['median'] = np.nanmedian(df['RT'])
    results['mean'] = np.nanmean(df['RT'])
    results['Std'] = np.nanstd(df['RT'])
    # no-brainer trials
    right = df['RT'][(df.RoptMag > df.LoptMag) &
                     (df.RoptProb > df.LoptProb)].values
    left = df['RT'][(df.LoptMag > df.RoptMag) &
                    (df.LoptProb > df.RoptProb)].values
    nobrainer = np.append(right, left)
    results['median nobrain'] = np.nanmedian(nobrainer)
    results['mean nobrain'] = np.nanmean(nobrainer)
    results['Std nobrain'] = np.nanstd(nobrainer)
    return results


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
        coefs[sub] = {}
        df = read_bids_logfile(sub, bidsdir)
        # save the maximum achieved points/gains in the experiment
        speed = stats_per_subject(sub, df=df)
        gain = np.max(df['points'])
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
                [(np.abs(scores[i][1:]) /
                  np.sum(np.abs(scores[i][1:])))[k]
                 for k in range(X.shape[1]) for i in range(n_splits) ]
                ),
            X.shape[1])
        # also average the beta coefficients without normalization
        coefs_pure = np.mean(scores[:, 1:], axis=0)
        # next, calculate average across folds for each parameter, as a label
        avg_coefs = np.mean(
            [np.abs(scores[i, 1:]) / np.sum(np.abs(scores[i, 1:]))
             for i in range(n_splits)],
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
        coefs[sub]['acc'] = avg_acc
        coefs[sub]['normed_coefs'] = list(avg_coefs)
        coefs[sub]['pure_coefs'] = list(coefs_pure)
        coefs[sub]['stats'] = speed
        coefs[sub]['gain'] = gain
        # save subject specific data
        fname = _construct_path([figdir, f'sub-{sub}', f'sub-{sub}_regression-results_behavior.json'])
        with open(fname, 'w') as f:
            json.dump(coefs[sub], f)
    # plot the reaction times
    plot_speed_stats(bidsdir, figdir)
    # correlate model accuracy with experiment performance
    accs = [coefs[k]['acc'] for k in coefs]
    gains = [coefs[k]['gain'] for k in coefs]
    corr = spearmanr(accs, gains)
    logging.info(f"The Spearman rank correlation between experiment "
                 f"performance (total gain) and model accuracy is "
                 f"{corr}")
    # plot overall importance of parameters affecting choice (average over subs)
    normed_averages = np.mean([coefs[c]['normed_coefs'] for c in coefs], axis=0)
    plot_relative_property_importance_group(normed_averages, labels, figdir)
    return coefs


def getmecoefs(est, X, y_true, **kwargs):
    """
    custom scorer to retrieve accuracies and coefficients from log. regression
    """
    y_pred = est.predict(X)
    acc = metrics.accuracy_score(y_true, y_pred)
    coefs = est.steps[1][1].coef_[0]
    return [acc] + list(coefs)


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
    fname = _construct_path(
        [figdir, f'sub-{sub}', f'logreg_stimulus-params-choice_subject-{sub}.svg'])
    logging.info(f'Saving a boxplot of parameter importance into {fname}.')
    plt.savefig(fname)
    plt.close('all')


def plot_relative_property_importance_group(coefs, labels, figdir):
    # make it a bar plot
    fig, ax = plt.subplots()
    ax.bar(labels, coefs)
    ax.set_ylabel('relative importance')
    ax.set_xlabel('stimulus property (Left or Right)')
    ax.set_title('Relative influence on choice by stimulus properties')
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.25)
    fig.tight_layout()
    fname = _construct_path([figdir, 'group', 'average_property_importance.svg'])
    fig.savefig(fname)


def plot_speed_stats(bidsdir, figdir='/tmp'):
    """Make a boxplot of aggregated reaction time statistics
     across subjects."""
    df = bigdf(bidsdir)
    # differentiate between brainer and no-brainer trials with a boolean column
    df['nobrainer'] = [
        ((df.RoptMag > df.LoptMag) & (df.RoptProb > df.LoptProb)) | \
        ((df.LoptMag > df.RoptMag) &(df.LoptProb > df.RoptProb))][0]
    df['nobrainer'] = df['nobrainer'].replace({True: 'No-brainer',
                                               False: 'Standard'})
    df = df.drop(columns=['trial_no', 'FixReqT', 'FixTime', 'orReqTime',
                          'orTime', 'LoptT','RoptT', 'FeedbackT', 'fix_onset',
                          'LoptOnset', 'or_onset', 'RoptOnset','response_onset',
                          'feedback_onset', 'trial_no.1', 'LoptProb', 'LoptMag',
                          'RoptProb', 'RoptMag', 'LoptRew', 'RoptRew', 'choice',
                          'points','pointdiff', 'timeoutflag', 'breaktrial',
                          'subject', 'pause_start', 'Empty_screen',
                          'second_delay_screen'])
    df = df.dropna()
    df['nobrainer'] = df['nobrainer'].astype('category')
    # plot overall RTs
    plt.figure()
    fig = sns.violinplot(data=df, x='RT')
    fig.set_xlabel('Reaction time in seconds')
    fig.set_title('Reaction times')
    fname = _construct_path(
        [figdir, 'group', 'memento_aggregate_reaction_times.svg'])
    fig.figure.savefig(fname)
    plt.figure()
    split = sns.violinplot(data=df, x='RT', y='nobrainer')
    split.set_xlabel('Reaction time in seconds')
    split.set_ylabel('Trial type')
    split.set_title('Reaction times ("no-brainer" trials versus standard trials)')
    fname = _construct_path(
        [figdir, 'group', 'memento_aggregate_reaction_times_nobrainer.svg'])
    split.figure.savefig(fname)




def regress_reinstatement_on_behavior(workdir,
                                      dimreduction,
                                      target,
                                      bidsdir,
                                      summary_metric='balanced accuracy'
                                      ):
    """
    For now this is work in progress.
    :param datadir:
    :param dimreduction:
    :param target:
    :param bidsdir:
    :param workdir:
    :return:
    """
    pathprefix = Path(workdir) / 'sub-0*'
    subs = [f[-3:] for f in glob(str(pathprefix))]

    for sub in subs:
        if dimreduction:
            fname = Path(workdir) / f'sub-{sub}' / f'{dimreduction}' / \
                f'sub-{sub}_decoding-scores_{target}.npy'
        else:
            fname = Path(workdir) / f'sub-{sub}' / \
                f'sub-{sub}_decoding-scores_{target}.npy'
        # read in confusion matrices
        scores = np.load(fname)
        # calculate the chosed summary metric from confusion matrices
        acrossclasses = np.asarray(
            [np.nanmean(get_metrics(c, metric=summary_metric))
             for score in scores
             for c in np.rollaxis(score, -1, 0)]).reshape(len(scores),
                                                          scores.shape[-1])
        # aggregate over cross-validation folds
        decoding = np.mean(acrossclasses, axis=0)
        behav_data = read_bids_logfile(sub, bidsdir)