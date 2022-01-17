"""
This script exists to describe, model, and analyze behavioral data.
Its mostly a placeholder for now.
Input by more experienced cognitive scientists is probably useful before I start
"""


import numpy as np
import pandas as pd
from glob import glob
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


def models():
    """
    TODO: Use this function for modelling the behavioral data. Ask Luca &
    others for advice maybe
    :return:
    """