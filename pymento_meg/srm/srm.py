"""
Module for fitting shared response models.
TODO:
Reduce epoch length to 6 seconds
OR reduce epoch length further, to potentially loose fewer trials

write all shared responses into a matrix, check for consistent correlation
pattern across subjects

"""


import mne
import logging
import random
import time

import numpy as np
import pandas as pd

from brainiak.funcalign import srm
from pathlib import Path
from scipy import stats
from textwrap import wrap
from pymento_meg.config import trial_characteristics
from pymento_meg.orig.behavior import read_bids_logfile
from pymento_meg.proc.epoch import get_stimulus_characteristics
from pymento_meg.srm.simulate import (
    transform_to_power,
    get_transformations
)
from pymento_meg.utils import _construct_path
import scipy.spatial.distance as sp_distance
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# set a seed to make train and test splits deterministic
random.seed(423)


def plot_global_field_power(epochs):
    """
    :return: Calculate and plot global field power, a measure of agreement of
    the signals picked up by all sensors across the entire scalp. Zero if all
    sensors have the same value at a given time point, non-zero if signals
    differ. GFP peaks may reflect “interesting” brain activity. Mathematically,
    the GFP is the population standard deviation across all sensors, calculated
    separately for every time point."""

    avg = epochs.average()
    gfp = avg.data.std(axis=0, ddof=0)
    # plot
    fig, ax = plt.subplots()
    ax.plot(avg.times, gfp * 1e6)
    ax.set(xlabel='Time (s)', ylabel='GFP (µV)', title='MEG')
    fname = TODO
    fig.savefig(fname)


def srm_with_spectral_transformation(subject,
                                     datadir,
                                     bidsdir,
                                     k=10,
                                     ntrain=140,
                                     ntest=100,
                                     modelfit='epochwise',
                                     ):
    """
    Fit a shared response model on data transformed into frequency space.
    This is the practical implementation of the work simulated in
    pymento_meg.srm.simulate.

    :param subject: list, all subjects to include
    :param ntrain: int, number of epochs to use in training
    :param ntest: int, number of epochs to use in testing
    :param datadir: str, path to directory with epoched data
    :param bidsdir: str, path to directory with bids data
    :param k: int, number of components used in SRM fitting
    :param modelfit: str, either 'epochwise', 'subjectwise', or 'trialorder'. Determines
     whether SRM is fit on all epochs as if they were subjects, subjectwise
     on averaged epochs, or subjectwise on articifial timeseries
    :return:
    """
    logging.warning("CAVE: I expect to operate on epochs starting with the 2nd "
                    "stimulation. Abort if it runs on the full 7s trial epochs")
    # read in epochs from all subjects. We don't need a special subset of the
    # data, and take the full timespan of each 3s epoch.

    # TODO: distinguish trials with positive and negative feedback
    fullsample, data = get_general_data_structure(subject=subject,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition='nobrain-brain',
                                                  timespan=[0, 300])

    # TODO: turn this into some sort of cross-validation?
    trainset, testset = train_test_set(fullsample,
                                       data,
                                       ntrain=ntrain,
                                       ntest=ntest,
                                       modelfit=modelfit)
    if modelfit == 'epochs':
        # transform  epochs to a time resolved and a spectral space. In order to
        # get components reflecting processes within 3 second epochs, loosen the
        # subject definition, and regard each individual epoch as a subject
        train_spectral, train_series = epochs_to_spectral_space(trainset)
        test_spectral, test_series = epochs_to_spectral_space(testset)
    elif modelfit == 'subjectwise':
        # Alternative: average epochs within subjects (in spectral space)
        train_spectral, train_series = \
            epochs_to_spectral_space(trainset, subjectwise=True)
        test_spectral, test_series = \
            epochs_to_spectral_space(testset, subjectwise=True)
    elif modelfit == 'trialtype':
        # the test and train data should be in a different format in this
        # condition, and require some downstream processing (concatenation)
        train_spectral, train_series = \
            concat_epochs_to_spectral_space(trainset, subjectwise=True)
        test_spectral, test_series = \
            concat_epochs_to_spectral_space(testset, subjectwise=True)

    else:
        logging.info(f"Unknown parameter {modelfit} for modelfit.")
        return

    # fit a shared response model on training data in spectral space
    model = shared_response([ts for k, ts in train_spectral.items()],
                            features=k)
    # transform the test data with it
    transformed = get_transformations(model, test_series, k)
    _plot_transformed_components(transformed, k, testset, adderror=False)



def get_transformations(model, test_series, comp):
    """
    Transform raw data into model space.
    :param model:
    :param raw:
    :param comp: number of components in the model
    :return:
    """
    transformations = {}
    for id, sub in enumerate(test_series.keys()):
        transformations[sub] = {}
        for k in range(comp):
            transformations[sub][k] = \
                [np.dot(model.w_[id].T, ts)[k] for ts in test_series[sub]]
    return transformations


def _get_mean_and_std_from_transformed(transformed, i, stderror=False):
    """Helper to get mean and std/sem vectors for a given component i from the
    transformed dict

    :param transformed: dict, contains transformed data
    :param i: component index
    :param stderror: bool, if True, return SEM instead of STD
    """
    mean = np.mean(np.asarray(
        [ts for sub in transformed.keys() for ts in transformed[sub][i]]),
        axis=0)
    # potentially change to standard error by dividing by np.sqrt(nepochs)

    if stderror:
        data = np.asarray(
            [ts for sub in transformed.keys() for ts in transformed[sub][i]]
        )
        std = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    else:
        # use standard deviation
        std = np.std(np.asarray(
        [ts for sub in transformed.keys() for ts in transformed[sub][i]]),
        axis=0)

    return mean, std


def _plot_helper(k,
                 suptitle,
                 name,
                 figdir,
                 palette='husl',
                 npalette=None,
                 figsize=(10, 20),
                 xlabel='samples',
                 ylabel='amplitude',
                 vline=None,
                 vlinelabel='response',
                 ):
    """
    Set up common plotting properties for SRM component plots to avoid code
    duplication.
    :param k: int, number of components.
    :param suptitle: str, title of the figure
    :param name: str, name under which the figure will be saved
    :param figdir: str, path to location to save figure under
    :param palette: str, name of a seaborn color map
    :param npalette: int, number of colors in the colormap
    :param figsize: tuple, specifies figure size
    :param xlabel: str, xlabel for the plot
    :param ylabel: str, ylabel for the plot
    :param vline: None or int, whether and where one vertical line will be drawn
    :param vlinelabel: str, label for a vline
    :return:
    """
    if npalette is None:
        npalette = k
    palette = sns.color_palette(palette, npalette)
    fig, ax = plt.subplots(k,
                           sharex=True,
                           sharey=True,
                           figsize=figsize)
    if k > 1:
        # multiple plots
        for a in ax:
            a.set(ylabel=ylabel)
            if vline is not None:
                a.axvline(vline,
                          color='black',
                          linestyle='dotted',
                          label=vlinelabel)
        ax[-1].set(xlabel=xlabel)
    else:
        ax.set(ylabel=ylabel)
        if vline is not None:
            ax.axvline(vline,
                      color='black',
                      linestyle='dotted',
                      label=vlinelabel)
        ax.set(xlabel=xlabel)
    fig.suptitle("\n".join(wrap(suptitle, 50)),
                 verticalalignment='top',
                 fontsize=10)
    fname = _construct_path(
        [
            Path(figdir),
            f"group",
            "meg",
            name,
        ]
    )
    return palette, fig, ax, fname


def _get_trial_indicators(transformed, data, type='choice'):
    """Query the metadata of the transformed data and report indices of trials
     corresponding to different properties"""

    if type == 'choice':
        # Get indices for trials with left and right choice
        i = 0
        left = []
        right = []
        for sub in transformed.keys():
            for epoch in data[sub]:
                if epoch['choice'] == 2:
                    right.append(i)
                elif epoch['choice'] == 1:
                    left.append(i)
                i += 1
        return left, right

    elif type == 'feedback':
        # Get trials with positive or negative feedback
        i = 0
        negative = []
        positive = []
        for sub in transformed.keys():
            for epoch in data[sub]:
                if np.isnan(epoch['pointdiff']):
                    negative.append(i)
                else:
                    positive.append(i)
                i += 1
        return negative, positive

    elif type == 'difficulty':
        # split between brainer and no-brainer trials
        i = 0
        brainer = []
        nobrainer = []
        for sub in transformed.keys():
            for epoch in data[sub]:
                if epoch['trial_type'] == 'brainer':
                    brainer.append(i)
                elif epoch['trial_type'] in (
                'nobrainer_right', 'nobrainer_left'):
                    nobrainer.append(i)
                i += 1
        return brainer, nobrainer


def _plot_raw_comparison(data, dataset, adderror=False, stderror=False, freq=1000,
                         window=0.5, figdir='/tmp'):
    """
    Do comparison plots with raw data
    :param data: dict, either train_series or test_series
    :param dataset: dict, either trainset or testset
    :return:
    """
    # first, not centered, averaged over all subjects, per sensor:
    d = np.asarray([np.mean(np.asarray(i), axis=0) for i in data.values()])
    mean = np.mean(d, axis=0)
    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Averaged raw signal, over all subjects, per sensor',
                     name=f"avg-signal_raw-sensors.png",
                     figdir=figdir,
                     npalette=1,
                     figsize=(10, 5)
                     )
    ax.plot(mean.T, label='averaged raw data, per sensor')
    fig.savefig(fname)
    # then averaged across sensors:
    moremean = np.mean(mean, axis=0)
    if stderror:
        std = np.std(mean, axis=0, ddof=1) / np.sqrt(mean.shape[0])
    else:
        std = np.std(mean, axis=0)
    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Averaged raw signal, over all subjects and sensors',
                     name=f"avg-signal_raw-avg.png",
                     figdir=figdir,
                     npalette=1,
                     figsize=(10, 5)
                     )
    ax.plot(moremean.T, label='averaged raw data', color=palette[0])
    if adderror:
        ax.fill_between(range(len(moremean)), moremean - std, moremean + std, alpha=0.4,
                           color=palette[0])
    fig.savefig(fname)
    # now response centered
    RT = [np.round(epoch['RT']*freq)
          for subject in data for epoch in dataset[subject]]
    # time window centered around the reaction
    win = window*freq
    d = []
    [d.extend(data[i]) for i in data.keys()]
    assert len(d) == len(RT)

    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Averaged raw signal, over all subjects, per sensor, response-locked',
                     name=f"avg-signal_raw-sensors_response-locked.png",
                     figdir=figdir,
                     npalette=1,
                     vline=win / 2,
                     figsize=(10, 5)
                     )
    # first, averaged over all subjects, per sensor
    # get all epochs
    centered_epochs = [d[idx][:, int(rt - win/2):int(rt + win/2)]
                      for idx, rt in enumerate(RT) if not np.isnan(rt)]
    # if an epoch does not have enough data (too short), don't use it
    d_long_enough = np.asarray([e for e in centered_epochs if e.shape[1] == win])
    # average over epochs
    avg_epochs = np.mean(d_long_enough, axis=0)
    # plot
    ax.plot(avg_epochs.T, label='averaged raw data (306 sensors)')
    fig.savefig(fname)

    # average over sensors
    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Averaged raw signal, over all subjects and sensors, response-locked',
                     name=f"avg-signal_raw-avg_response-locked.png",
                     figdir=figdir,
                     npalette=1,
                     vline=win / 2,
                     figsize=(10, 5),
                     )
    avg = np.mean(avg_epochs, axis=0)
    ax.plot(avg, label='averaged raw data (across subjects and sensors',
            color=palette[0])
    if stderror:
        std = np.std(avg_epochs, axis=0, ddof=1) / np.sqrt(avg_epochs.shape[0])
    else:
        std = np.std(avg_epochs, axis=0)
    if adderror:
        ax.fill_between(range(len(avg)), avg - std, avg + std, alpha=0.4,
                           color=palette[0])
    fig.savefig(fname)
    # and now for left and right
    # now response-locked for left and right
    left, right = _get_trial_indicators(dataset, dataset, type='choice')
    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Average raw signal, response-locked, left vs. right',
                     name=f"avg-signal_raw-avg_response-locked_leftvsright.png",
                     figdir=figdir,
                     npalette=2,
                     vline=win/2,
                     figsize=(10, 5)
                     )
    b = 0
    for choice, ids in [('left', left), ('right', right)]:
        color_idx = b
        centered_epochs = [d[idx][:, int(rt - win / 2):int(rt + win / 2)]
                           for idx, rt in enumerate(RT) if idx in ids and
                           not np.isnan(rt)]
        # if an epoch does not have enough data (too short), don't use it
        d_long_enough = np.asarray(
            [e for e in centered_epochs if e.shape[1] == win])
        # average twice:
        avg_epochs = np.mean(d_long_enough, axis=0)
        avg = np.mean(avg_epochs, axis=0)
        ax.plot(avg, label=f'{choice} choice', color=palette[b])
        if adderror:
            if stderror:
                std = np.std(avg_epochs, axis=0, ddof=1) / \
                      np.sqrt(avg_epochs.shape[0])
            else:
                std = np.std(avg_epochs, axis=0)
            ax.fill_between(range(len(std)), avg - std, avg + std,
                               alpha=0.3,
                               color=palette[color_idx])
        b += 1
    plt.legend(loc="upper right", prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)


def _plot_transformed_components(transformed,
                                 k,
                                 data,
                                 adderror=False,
                                 figdir='/tmp',
                                 stderror=False,
                                 modelfit=None
                                 ):
    """
    For transformed data containing the motor response/decision, create a range
    of plots illustrating the data in shared space.
    :transformed: dict, raw data transformed into shared space
    :param data: either trainset or testset
    :param k: int, number of features in the model
    :param adderror: bool, whether to add the standard deviation around means
    :param figdir: str, Path to a place to save figures
    :param stderror: bool, if true, SEM is used instead of std
    :return:
    """
    # plot transformed components:
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Averaged signal in shared space, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
                     figdir=figdir,
                     npalette=k,
                     )
    for i in range(k):
        mean, std = _get_mean_and_std_from_transformed(transformed, i,
                                                       stderror=stderror)
        ax[i].plot(mean, color=palette[i], label=f'component {i+1}')
        if adderror:
            # to add standard deviations around the mean. We didn't find expected
            # congruency/reduced variability in those plots.
            ax[i].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.4,
                               color=palette[i])
    for a in ax:
        a.legend(loc='upper right',
                 prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)

    if modelfit == 'trialtype':
        # the testset of this data is differently structured and the code below
        # would error

        _plot_transformed_components_by_trialtype(transformed, k, data,
                                                  adderror=True,
                                                  stderror=True,
                                                  plotting='all'
                                                  )
        return

    # Plot transformed data component-wise, but for left and right epochs
    # separately.
    left, right = _get_trial_indicators(transformed, data, type='choice')
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space for left & right choices, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp_leftvsright.png",
                     figdir=figdir,
                     npalette=k*2,
                     )
    b = 0
    for choice, ids in [('left', left), ('right', right)]:
        for i in range(k):
            comp = []
            for sub in transformed.keys():
                comp.extend(transformed[sub][i])
            d = np.asarray([l for idx, l in enumerate(comp) if idx in ids])
            color_idx = b + i
            mean = np.mean(d, axis=0)
            if stderror:
                std = np.std(d, axis=0, ddof=1) / np.sqrt(d.shape[0])
            else:
                std = np.std(d, axis=0, ddof=0)
            ax[i].plot(mean, color=palette[color_idx],
                       label=f'component {i+1}, {choice} choice')
            if adderror:
                ax[i].fill_between(range(len(mean)),
                                   mean-std,
                                   mean+std,
                                   alpha=0.3,
                                   color=palette[color_idx])
        b = k
    for a in ax:
        a.legend(loc="upper right", prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)


def _plot_fake_transformed(order,
                           data,
                           title,
                           name,
                           transformed,
                           figdir,
                           k,
                           stderror,
                           adderror,
                           bychoice=False):
    """Aggregate data with trial structure into a temporary structure"""
    # get ids for each subject and trialtype - the number of trialtypes differs
    # between subjects, and we need the ids to subset the consecutive list of
    # them in 'transformed'
    ids = {}
    for sub in transformed.keys():
        ids[sub] = {}
        i = 0
        # the trialtypes are consecutive in this order
        for trial in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            ids[sub][trial] = (i, i+len(data[sub][trial]))
            i += len(data[sub][trial])
    if not bychoice:
        palette, fig, ax, fname = \
            _plot_helper(k,
                         suptitle=title,
                         name=name,
                         figdir=figdir,
                         npalette=len(order),
                         )
        # don't group trials according to if they were eventually selected
        for colorid, trials in enumerate(order):
            # we group several trialtypes together
            tmp_transformed = {}
            for sub in transformed.keys():
                tmp_transformed[sub] = {}
                for trial in trials:
                    # get indices of specific trial type
                    idx = ids[sub][trial]
                    assert idx[0] < idx[1]
                    for c in range(k):
                        tmp_transformed[sub].setdefault(c, []).extend(
                            transformed[sub][c][idx[0]:idx[1]])
            # plot the component
            for i in range(k):
                mean, std = _get_mean_and_std_from_transformed(tmp_transformed,
                                                               i,
                                                               stderror=stderror
                                                               )
                lab = f'{label[colorid] if label else trials}, k={i+1}'
                ax[i].plot(mean,
                           color=palette[colorid],
                           label=lab)
                if adderror:
                    ax[i].fill_between(range(len(mean)), mean - std, mean + std,
                                       alpha=0.1,
                                       color=palette[colorid])
            # Finally, add the legend.
            for a in ax:
                a.legend(loc='upper right',
                         prop={'size': 6})
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(fname)

    else:
        # group trials by whether they were eventually chosen
        right = {}
        left = {}
        for sub in transformed.keys():
            right[sub] = []
            left[sub] = []
            i = 0
            for trial in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                for epoch in data[sub][trial]:
                    if epoch['choice'] == 2:
                        right[sub].append(i)
                    elif epoch['choice'] == 1:
                        left[sub].append(i)
                    i += 1

        palette, fig, ax, fname = \
            _plot_helper(k,
                         suptitle=title,
                         name=name,
                         figdir=figdir,
                         npalette=len(order),
                         palette='rocket'
                         )

        palette2 = sns.color_palette('mako', len(order))
        palettes = [palette, palette2]
        for colorid, choiceids in enumerate([left, right]):
            cid = 0
            for trial in order:
                tmp_transformed = {}
                for sub in transformed.keys():
                    tmp_transformed[sub] = {}
                    for t in trial:
                        id = ids[sub][t]
                        assert id[0] < id[1]
                        for c in range(k):
                            data = [d for idx, d in enumerate(transformed[sub][c])
                                    if idx in choiceids[sub] and (id[1] <= idx >= id[0])]
                            tmp_transformed[sub].setdefault(c, []).extend(data)

                for comp in range(k):
                    mean, std = _get_mean_and_std_from_transformed(tmp_transformed,
                                                                   comp,
                                                                   stderror=stderror
                                                                   )
                    if label:
                        tid = order.index(trial)
                        lab = f'{colorid} choice, {label[tid]}, k={comp + 1}'
                    else:
                        lab = f'{colorid} choice,  trial {trial}, k={comp + 1}'
                    ax[comp].plot(mean,
                                 color=palettes[colorid][cid],
                                 label=lab)
                    if adderror:
                        ax[comp].fill_between(range(len(mean)), mean - std, mean + std,
                                           alpha=0.1,
                                           color=palettes[colorid][cid])
                cid += 1
                # Finally, add the legend.
            for a in ax:
                a.legend(loc='upper right',
                         prop={'size': 6})
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(fname)

    return fig, ax, fname


def _plot_transformed_components_by_trialtype(transformed,
                                              k,
                                              data,
                                              adderror=False,
                                              stderror=False,
                                              figdir='/tmp',
                                              plotting='all'
                                              ):
    """

    :param transformed:
    :param k:
    :param data:
    :param adderror:
    :param stderror:
    :param figdir:
    :param plotting:
    :return:
    """
    # define general order types
    magnitude_order = [('A', 'B'), ('C', 'D'), ('E', 'F', 'G'), ('H', 'I')]
    magnitude_labels = [('0.5 reward'), ('1 reward'), ('2 rewards'), ('4 reward')]
    trialorder = [('A'), ('B'), ('C'), ('D'), ('E'), ('F'), ('G'), ('H'), ('I')]
    probability_order = [('E', 'H'), ('C', 'F', 'I'), ('A', 'G'), ('B', 'D')]
    probability_labels = [('10% chance'), ('20% change'), ('40% chance'), ('80% chance')]
    exceptedvalue_order = [('A', 'C', 'E'), ('B', 'F', 'H'), ('D', 'G', 'I')]
    expectedvalue_labels = [('0.2 EV'), ('0.4 EV'), ('0.8 EV')]
    if plotting in ('puretrialtype', 'all'):
        fig, ax, fname = _plot_fake_transformed(
            order=trialorder,
            data=data,
            title="Transformed components, per trial type",
            name=f"trialtype-wise_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
            )

    if plotting in ('magnitude', 'all'):
        # plot according to magnitude bins
        fig, ax, fname = _plot_fake_transformed(
            order=magnitude_order,
            data=data,
            label=magnitude_labels,
            title="Transformed components, with trials grouped into magnitude bins",
            name=f"trialtype-magnitude_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
            )

    if plotting in ('probability', 'all'):
        # plot according to probability bins
        fig, ax, fname = _plot_fake_transformed(
            order=probability_order,
            data=data,
            label=probability_labels,
            title="Transformed components, with trials grouped into probability bins",
            name=f"trialtype-probability_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
            )

    if plotting in ('expectedvalue', 'all'):
        # plot according to expected value
        fig, ax, fname = _plot_fake_transformed(
            order=exceptedvalue_order,
            data=data,
            label=expectedvalue_labels,
            title="Transformed components, with trials grouped into expected value bins",
            name=f"trialtype-exp-value_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
        )
    if plotting in ('eventualchoice', 'all'):
        # plot trials that were eventually chosen versus trials that weren't
        # now response-locked for left and right
        right = {}
        left = {}
        for sub in transformed.keys():
            right[sub] = []
            left[sub] = []
            i = 0
            for trial in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                for epoch in data[sub][trial]:
                    if epoch['choice'] == 2:
                        right[sub].append(i)
                    elif epoch['choice'] == 1:
                        left[sub].append(i)
                    i += 1

    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Transformed components, with trials grouped by eventual response',
                     name=f'event-choice_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png',
                     figdir=figdir,
                     npalette=2,
                     )

    for colorid, ids in enumerate([left, right]):
        tmp_transformed = {}
        for sub in transformed.keys():
            tmp_transformed[sub] = {}
            for c in range(k):
                tmp_transformed[sub][c] = \
                    [d for idx, d in enumerate(transformed[sub][c])
                     if idx in ids[sub]]
        # plot the component
        for i in range(k):
            mean, std = _get_mean_and_std_from_transformed(tmp_transformed,
                                                           i,
                                                           stderror=stderror
                                                           )
            ax[i].plot(mean,
                       color=palette[colorid],
                       label=f'trial choice {colorid}, component {i + 1}')
            if adderror:
                ax[i].fill_between(range(len(mean)), mean - std, mean + std,
                                   alpha=0.1,
                                   color=palette[colorid])
        # Finally, add the legend.
        for a in ax:
            a.legend(loc='upper right',
                     prop={'size': 6})
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fname)

    if plotting in ('magnitude-by-choice', 'all'):
        # plot according to magnitude bins
        fig, ax, fname = _plot_fake_transformed(
            order=magnitude_order,
            data=data,
            label=magnitude_labels,
            title="Transformed components, with trials grouped into magnitude bins by eventual choice",
            name=f"trialtype-magnitude-bychoice_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            bychoice=True
        )

    if plotting in ('probability-by-choice', 'all'):
        # plot according to magnitude bins
        fig, ax, fname = _plot_fake_transformed(
            order=probability_order,
            data=data,
            label=probability_labels,
            title="Transformed components, with trials grouped into probability bins by eventual choice",
            name=f"trialtype-probability-bychoice_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='choice'
        )

    if plotting in ('expected-value-by-choice', 'all'):
        # plot according to expected value bins
        fig, ax, fname = _plot_fake_transformed(
            order=exceptedvalue_order,
            data=data,
            label=expectedvalue_labels,
            title="Transformed components, with trials grouped into expected value bins by eventual choice",
            name=f"trialtype-expectedvalue-bychoice_avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            bychoice=True
        )




def _plot_transformed_components_centered(transformed,
                                          k,
                                          data,
                                          adderror=False,
                                          window=0.4,
                                          freq=100,
                                          figdir='/tmp',
                                          stderror=False
                                          ):
    """
    For transformed data containing the motor response/decision, create a range
    of plots illustrating the data in shared space.
    :transformed: dict, raw data transformed into shared space
    :param data: either trainset or testset
    :param k: int, number of features in the model
    :param adderror: bool, whether to add the standard deviation around means
    :param window: float, number of seconds centered around a decision to plot
    :param freq: int, sampling frequency of the raw data
    :param figdir: str, Path to a place to save figures
    :param stderror: bool, if true, SEM is used instead of std
    :return:
    """
    # plots centered around reaction time
    RT = [np.round(epoch['RT']*freq)
          for subject in data for epoch in data[subject]]
    # time window centered around the reaction
    win = window*freq

    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space, response-locked, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp_response-locked.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win/2
                     )

    for i in range(k):
        comp = []
        for sub in transformed.keys():
            comp.extend(transformed[sub][i])

        d = [comp[idx][int(rt - win/2):int(rt + win/2)]
             for idx, rt in enumerate(RT) if not np.isnan(rt)]
        # if an epoch does not have enough data (too short), don't use it
        d_long_enough = np.asarray([lst for lst in d if len(lst) == win])
        mean = np.mean(d_long_enough, axis=0)
        ax[i].plot(mean, color=palette[i], label=f'component {i+1}')
        if adderror:
            if stderror:
                std = np.std(d_long_enough, axis=0, ddof=1) / \
                      np.sqrt(d_long_enough.shape[0])
            else:
                std = np.std(d_long_enough, axis=0)
            ax[i].fill_between(range(len(std)), mean-std, mean+std, alpha=0.3,
                               color=palette[i])
    for a in ax:
        a.legend(loc="upper right", prop={'size': 6})
    plt.legend(loc="upper right", prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)

    # now response-locked for left and right
    left, right = _get_trial_indicators(transformed, data, type='choice')
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space, response-locked, left vs. right, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp_response-locked_leftvsright.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win/2
                     )
    b = 0
    for choice, ids in [('left', left), ('right', right)]:
        for i in range(k):
            color_idx = b + i
            comp = []
            for sub in transformed.keys():
                comp.extend(transformed[sub][i])
            d = [comp[idx][int(rt - win/2):int(rt + win/2)] for idx, rt in
                 enumerate(RT) if idx in ids and not np.isnan(rt)]
            # if an epoch does not have enough data (too short), don't use it
            d_long_enough = np.asarray([lst for lst in d if len(lst) == win])
            mean = np.mean(d_long_enough, axis=0)
            ax[i].plot(mean, color=palette[color_idx],
                       label=f'component {i+1}, {choice} choice')
            if adderror:
                if stderror:
                    std = np.std(d_long_enough, axis=0, ddof=1) / \
                          np.sqrt(d_long_enough.shape[0])
                else:
                    std = np.std(d_long_enough, axis=0)
                ax[i].fill_between(range(len(std)), mean - std, mean + std,
                                   alpha=0.3,
                                   color=palette[color_idx])
        b = k
    for a in ax:
        a.legend(loc="upper right", prop={'size': 6})
    plt.legend(loc="upper right", prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)

    # split by difficulty (brainernobrainer)
    brainer, nobrainer = _get_trial_indicators(transformed, data,
                                               type='difficulty')
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space, response-locked, brainer vs nobrainer, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp_response-locked_brainervsnobrainer.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win/2
                     )
    b = 0

    for choice, ids in [('brainer', brainer), ('nobrainer', nobrainer)]:
        for i in range(k):
            color_idx = b + i
            comp = []
            for sub in transformed.keys():
                comp.extend(transformed[sub][i])

            d = [comp[idx][int(rt - win/2):int(rt + win/2)] for idx, rt in
                 enumerate(RT) if idx in ids and not np.isnan(rt)]
            # if an epoch does not have enough data (too short), don't use it
            d_long_enough = np.asarray([lst for lst in d if len(lst) == win])
            mean = np.mean(d_long_enough, axis=0)
            ax[i].plot(mean, color=palette[color_idx],
                       label=f'component {i+1}, {choice} trials')
            if adderror:
                if stderror:
                    std = np.std(d_long_enough, axis=0, ddof=1) / \
                          np.sqrt(d_long_enough.shape[0])
                else:
                    std = np.std(d_long_enough, axis=0)
                ax[i].fill_between(range(len(std)), mean - std, mean + std,
                                   alpha=0.3,
                                   color=palette[color_idx])
        b = k
    for a in ax:
        a.legend(loc="upper right", prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)

    # positive versus negative feedback
    negative, positive = _get_trial_indicators(transformed, data,
                                               type='feedback')
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space, response-locked, pos vs neg feedback, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per-comp_response-locked_feedback.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win/2
                     )
    b = 0
    for choice, ids in [('negative feedback', negative),
                        ('positive feedback', positive)]:
        for i in range(k):
            color_idx = b + i
            comp = []
            for sub in transformed.keys():
                comp.extend(transformed[sub][i])

            d = [comp[idx][int(rt - win / 2):int(rt + win / 2)] for idx, rt in
                 enumerate(RT) if idx in ids and not np.isnan(rt)]
            # if an epoch does not have enough data (too short), don't use it
            d_long_enough = np.asarray([lst for lst in d if len(lst) == win])
            mean = np.mean(d_long_enough, axis=0)
            ax[i].plot(mean, color=palette[color_idx],
                       label=f'component {i+1}, {choice}')
            if adderror:
                if stderror:
                    std = np.std(d_long_enough, axis=0, ddof=1) / \
                          np.sqrt(d_long_enough.shape[0])
                else:
                    std = np.std(d_long_enough, axis=0)
                ax[i].fill_between(range(len(std)), mean - std, mean + std,
                                   alpha=0.3,
                                   color=palette[color_idx])
        b = k
    for a in ax:
        a.legend(loc="upper right", prop={'size': 6})
    ax[-1].set(xlabel='samples')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)

    return


def concat_epochs_to_spectral_space(data, subjectwise=False, shorten=False):
    """
    Transform epoch data organized by trial features into spectral space.
    Average within subject and feature, and concatenate within subject over
    features. Takes a nested dictionary with n subjects as keys and features as
    subkeys, transforms each subjects epochs into spectral space, returns each
    epoch as if it was its own subject
    :param data: nested dict, top level keys are subjects, lower level dicts are
     trial data of one epoch
    :param shorten: None or tuple, if given, the data will be subset in th
     range of the tuple. The tuple needs to have a start and end index in Hz.
    :return:
    series: list of lists
    """
    subjects = data.keys()
    series_spectral = {}
    series_time = {}
    for sub in subjects:
        subject_spectral = []
        subject_time = []
        for condition in data[sub]:
            epochs = [info['normalized_data'] for info in data[sub][condition]]
            if shorten:
                # subset the data to the range given by 'shorten'
                epochs = [epo[:, shorten[0]:shorten[1]] for epo in epochs]
            spectral_series = [transform_to_power(e) for e in epochs]
            assert spectral_series[0].shape[0] == epochs[0].shape[0] == 306
            # average over epochs.
            spectral_series = np.mean(np.asarray(spectral_series), axis=0)
            assert spectral_series.shape[0] == epochs[0].shape[0] == 306
            # append the condition to concatenate all trialtypes per subject
            # TODO check if the dimensionality matches. Its is a list with 24300
            # arrays of dim 306 so far
            subject_spectral.extend(spectral_series.T)
            # TODO, does not yet make sense
            subject_time.extend(epochs)
        series_spectral[sub] = np.asarray(subject_spectral).T
        series_time[sub] = subject_time
    assert len(series_spectral) == len(series_time) == len(subjects)
    return series_spectral, series_time


def epochs_to_spectral_space(data, subjectwise=False):
    """
    Transform epoch data into spectral space. Takes a dictionary with n subjects
    as keys, transforms each subjects epochs into spectral space, returns each
    epoch as if it was its own subject
    :param data: nested dict, top level keys are subjects, lower level dicts are
     trial data of one epoch
    :return:
    series: list of lists
    """
    subjects = data.keys()
    series_spectral = {}
    series_time = {}
    for sub in subjects:
        epochs = [info['normalized_data'] for info in data[sub]]
        spectral_series = [transform_to_power(e) for e in epochs]
        assert spectral_series[0].shape[0] == epochs[0].shape[0] == 306
        if subjectwise:
            # average over epochs. This results in one train epoch only.
            spectral_series = np.mean(np.asarray(spectral_series), axis=0)
            assert spectral_series.shape[0] == epochs[0].shape[0] == 306
        series_spectral[sub] = spectral_series
        series_time[sub] = epochs
    assert len(series_spectral) == len(series_time) == len(subjects)
    return series_spectral, series_time


def train_test_set(fullsample, data, ntrain=140, ntest=100, modelfit=None):
    """
    Split data into ntrain trainings data and ntest test data, and return them
    as a dictionary. If a particular modelfit is specified, ntrain and ntest are
    ignored, and instead, data are returned as a nested dictionary according to
    the specified experiment featured, with half of all available data in the
    training set and the other half in the testset.
    :param fullsample:
    :param data:
    :param ntrain:
    :param ntest:
    :param modelfit:
    :return:
    """
    testset = {}
    trainset = {}
    subjects = fullsample.keys()
    if modelfit == 'trialtype':
        logging.info(f"Splitting the data into training and testing epochs "
                     f"based on trial type.")
        # because we want to concatenate artificial time series, we let go of
        # ntrain and ntest specification and instead take all data we have
        trialorder = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        for sub in subjects:
            testset[sub] = {}
            trainset[sub] = {}
            for trialtype in trialorder:
                testset[sub][trialtype] = []
                trainset[sub][trialtype] = []
                trials = [i for i in data if (i['subject'] == sub and
                                              i['Lchar'] == trialtype)]
                # take half of the trials for training, half of them for testing
                ntest = ntrain = int(len(trials) / 2)
                shuffled = random.sample(trials, len(trials))
                trainset[sub][trialtype].extend(shuffled[:ntrain])
                testset[sub][trialtype].extend(shuffled[-ntest:])
        # return a nested dictionary. Later functions will need to disect it
        return trainset, testset

    else:
        logging.info(f"Splitting the data into {ntrain} training epochs and "
                     f"{ntest} testing epochs.")
        # just take the trials "as is"
        for subject in subjects:
            train = []
            test = []
            trials = [i for i in data if i['subject'] == subject]
            # we need to have at least as many trials per condition as the
            # sum of training data and testing data
            assert len(trials) >= ntrain + ntest
            # for each trialtype, pick n random trials for training,
            # shuffle the list with a random seed to randomize trial order
            shuffled = random.sample(trials, len(trials))
            train.extend(shuffled[:ntrain])
            # pick test trials from the other side of the shuffled list
            test.extend(shuffled[-ntest:])
            assert all([i not in test for i in train])
            testset[subject] = test
            trainset[subject] = train
            assert len(testset[subject]) == ntest
            assert len(trainset[subject]) == ntrain
    return trainset, testset


def plot_model_basis_topographies(datadir, model, figdir):
    """
    Take the subject-specific basis for each component of a trained SRM model
    and plot their topography.
    :return:
    """
    # use real data to create a fake evoked structure
    fname = Path(datadir) / f'sub-001/meg' / \
            f'sub-001_task-memento_proc-sss_meg-1.fif'
    raw = mne.io.read_raw_fif(fname)
    # drop all non-meg sensors from the info object
    picks = raw.info['ch_names'][3:309]
    raw.info.pick_channels(picks)

    for subject in range(len(model.w_)):
        basis = model.w_[subject]
        k = basis.shape[1]
        fig, ax = plt.subplots(1, k)
        for c in range(k):
            # plot transformation matrix
            data = basis[:, c].reshape(-1, 1)
            fake_evoked = mne.EvokedArray(data, raw.info)
            fig = fake_evoked.plot_topomap(times=0,
                                           title=f'Subject {subject+1}',
                                           colorbar=False,
                                           axes=ax[c], size=2
                                           )
        fname = _construct_path([
            Path(figdir),
            "group",
            "meg",
            f"viz-model-{k}_comp-{c}_sub-{subject+1}.png"])
        fig.savefig(fname)



def get_general_data_structure(subject,
                               datadir,
                               bidsdir,
                               condition,
                               timespan):
    """
    Retrieve trial-wise data and its characteristics into a nested dictionary
    and the linearization of this structure, a list of dictionaries.

    :param subject: str, subject identifier
    :param datadir: str, path to data with epochs
    :param bidsdir: str, path to bids directory
    :param condition: str, an identifier based on which trials can be split.
    Possible values: 'left-right' (left versus right option choice),
    'nobrain-brain' (no-brainer versus brainer trials)
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), or 'fulltrial' (entire 7second epoch), 'secondstim',
    'delay'.
    :return:
    fullsample: dict; holds data of all subjects (keys are subjects)
    data: list of dicts, each entry is a dict with one epoch of one subject.
    """
    # initialize a dict to hold data over all subjects in the model
    fullsample = {}
    if not isinstance(subject, list):
        # we may want to combine multiple subject's data.
        subject = [subject]

    for sub in subject:
        fname = Path(datadir) / f'sub-{sub}/meg' / \
                f'sub-{sub}_task-memento_cleaned_epo.fif'
        logging.info(f'Reading in cleaned epochs from subject {sub} '
                     f'from path {fname}.')

        epochs = mne.read_epochs(fname)
        freq = int(epochs.info['sfreq'])
        logging.info(f'The data has a frequency of {freq}Hz')
            # after initial preprocessing, they are downsampled to 200Hz.
            # Downsample further to 100Hz
            #epochs.resample(sfreq=100, verbose=True)
        #assert epochs.info['sfreq'] == 100
        # read the epoch data into a dataframe
        df = epochs.to_data_frame()

        # use experiment logdata to build a data structure with experiment
        # information on every trial
        trials_to_trialtypes, epochs_to_trials = \
            _find_data_of_choice(epochs=epochs,
                                 subject=sub,
                                 bidsdir=bidsdir,
                                 condition=condition,
                                 df=df)

        all_trial_info = combine_data(df=df,
                                      sub=sub,
                                      trials_to_trialtypes=trials_to_trialtypes,
                                      epochs_to_trials=epochs_to_trials,
                                      bidsdir=bidsdir,
                                      timespan=timespan)
        # add trial order information to the data structure
        all_trial_info = add_trial_types(subject=sub,
                                         bidsdir=bidsdir,
                                         all_trial_info=all_trial_info)

        # append single subject data to the data dict of the sample
        fullsample[sub] = all_trial_info
    # aggregate the data into a list of dictionaries
    data = []
    for subject, trialinfo in fullsample.items():
        for trial_no, info in trialinfo.items():
            info['subject'] = subject
            info['trial_no'] = trial_no
            data.append(info)

    return fullsample, data


def _create_splits_from_left_and_right_stimulation(subjects,
                                                   leftdata,
                                                   rightdata,
                                                   trialorder,
                                                   ntrain,
                                                   ntest):
    """
    Create a number of data aggregates, split into test and train sets, both in
    raw form and in averaged form.
    :param subjects: list of subjects included in the data
    :param leftdata: dict, contains the data from the left stimulation
    :param rightdata: dict, contains the data from the right stimulation
    :param ntrain: int, number of trials to use in training
    :param ntest: int, number of trials to use in testing
    :param trialorder: list, order in which trials should be concatenated
    :return: nested dict, with keys indicating original or averaged, then train
    or test, and then left or fullseries
    """

    train_data = {}
    mean_train_data = {}
    test_data = {}
    mean_test_data = {}
    for data, position, index in [(leftdata, 'left', 'Lchar'),
                                  (rightdata, 'right', 'Rchar')]:
        train_data[position] = {}
        mean_train_data[position] = {}
        test_data[position] = {}
        mean_test_data[position] = {}
        for sub in subjects:
            train = []
            mean_train = []
            test = []
            mean_test = []
            for trialtype in trialorder:
                trials = [i for i in data if (i['subject'] == sub and
                                              i[index] == trialtype)]
                # we need to have at least as many trials per condition as the
                # sum of training data and testing data
                assert len(trials) >= ntrain + ntest
                # for each trialtype, pick n random trials for training,
                # shuffle the list with a random seed to randomize trial order
                shuffled = random.sample(trials, len(trials))
                train.extend(shuffled[:ntrain])
                # pick test trials from the other side of the shuffled list
                test.extend(shuffled[-ntest:])
                assert all([i not in test for i in train])

                # average data across trials of trialtype for noise reduction
                mean = np.mean(
                    [t['normalized_data'] for t in shuffled[:ntrain]],
                    axis=0)
                assert mean.shape[0] == 306
                mean_train.append(mean)
                # do the same for test trials
                mean = np.mean(
                    [t['normalized_data'] for t in shuffled[-ntest:]],
                    axis=0)
                assert mean.shape[0] == 306
                mean_test.append(mean)
            # after looping through 9 trial types make sure we have the expected
            # amount of data
            assert len(train) == 9 * ntrain
            assert len(test) == 9 * ntest
            assert len(mean_train) == 9
            assert len(mean_test) == 9
            train_data[position][sub] = train
            mean_train_data[position][sub] = mean_train
            test_data[position][sub] = test
            mean_test_data[position][sub] = mean_test
    # join the data from left and right into nested lists. Also make a only-left
    # series
    mean_train_data_fullseries = []
    mean_train_data_leftseries = []
    mean_test_data_fullseries = []
    mean_test_data_leftseries = []
    for sub in subjects:
        left = concatenate_means(mean_train_data['left'][sub])
        right = concatenate_means(mean_train_data['right'][sub])
        assert left.shape == right.shape
        mean_train_data_fullseries.append(concatenate_means([left, right]))
        mean_train_data_leftseries.append(left)
        # also for test data
        left = concatenate_means(mean_test_data['left'][sub])
        right = concatenate_means(mean_test_data['right'][sub])
        assert left.shape == right.shape
        mean_test_data_fullseries.append(concatenate_means([left, right]))
        mean_test_data_leftseries.append(left)
    assert len(mean_train_data_fullseries) == \
           len(mean_train_data_leftseries) == \
           len(subjects) == \
           len(mean_test_data_leftseries) == \
           len(mean_test_data_fullseries)

    train_data_fullseries = []
    train_data_leftseries = []
    for sub in subjects:
        left = concatenate_data(train_data['left'][sub])
        right = concatenate_data(train_data['right'][sub])
        assert left.shape == right.shape
        train_data_fullseries.append(concatenate_means([left, right]))
        train_data_leftseries.append(left)
    assert len(train_data_fullseries) == \
           len(train_data_leftseries) == \
           len(subjects)

    test_data_fullseries = []
    test_data_leftseries = []
    for sub in subjects:
        left = concatenate_data(test_data['left'][sub])
        right = concatenate_data(test_data['right'][sub])
        assert left.shape == right.shape
        test_data_fullseries.append(concatenate_means([left, right]))
        test_data_leftseries.append(left)
    assert len(test_data_fullseries) == \
           len(test_data_leftseries) == \
           len(subjects)

    # return all this data as a dict
    results = {'original_trials':
                   {'test':
                        {'left': test_data_leftseries,
                         'full': test_data_fullseries
                        },
                    'train':
                        {'left': train_data_leftseries,
                         'full': train_data_fullseries
                        }
                    },
               'averaged_trials':
                   {'test':
                        {'left': mean_test_data_leftseries,
                         'full': mean_test_data_fullseries
                        },
                    'train':
                        {'left': mean_train_data_leftseries,
                         'full': mean_train_data_fullseries
                        }
                    }
               }

    return results


def test_and_train_split(datadir,
                         bidsdir,
                         figdir,
                         subjects=None,
                         ntrain=15,
                         ntest=15,
                         timespan=None):
    """
    Create artificially synchronized time series data. In these artificially
    synchronized timeseries, N=ntrain trials per trial type (unique probability-
    magnitude combinations of the first stimulus) are first concatenated in
    blocks. This is done for a train set and a test set.
    Then, trials within each trial type block are averaged for noise reduction.

    # TODO: Maybe use training data of shorter length (e.g., visual stim), and
    # test data of longer length/different trial segments
    # TODO: use the first 500 ms

    # TODO: threshold für die mean trial distance matrices
    # TODO: sanity check mit test data
    # TODO: Transformation von test data mit srm vom training
    :param datadir:
    :param bidsdir:
    :param figdir:
    :param subjects: set of str of subject identifiers. Default are subjects
    with the largest number of suitable trial data (30+ per condition)
    :param ntrain: int, number of trials to put in training set
    :param ntest: int, number of trials to put in test set
    :param timespan: dict, a specification of the time span to use for
     subsetting the data. When it isn't a string identifier from combine_data,
     it can be a list with start and end sample in 100Hz, e.g. {'left': [0, 70],
     'right': [270, 340]}
    :return:
    train_series: list of N=subjects lists artificial time series data,
    ready for SRM
    mean_train_series: list of N=subjects lists averaged artificial time series
     data, ready for SRM
    training_set_left: dict; overview of trials used as training and test data
     for each subject
    training_set_right:  dict; overview of trials used as training and test data
     for each subject
    """
    if subjects is None:
        # default to a list of subjects with the most good trial events
        subjects = ['011', '012', '014', '016', '017',
                    '018', '019', '020', '022']
    if timespan is None:
        timespan = {'left': 'firststim',
                    'right': 'secondstim'}
    timespan_left = timespan['left']
    timespan_right = timespan['right']
    triallength = 70 if isinstance(timespan_left, str) \
        else timespan_left[1] - timespan_left[0]
    logging.info(f"Setting trial length to {triallength}")
    # first stimulus data
    leftsample, leftdata = get_general_data_structure(subject=subjects,
                                                      datadir=datadir,
                                                      bidsdir=bidsdir,
                                                      condition='left-right',
                                                      timespan=timespan_left)
    # second stimulus data
    rightsample, rightdata = get_general_data_structure(subject=subjects,
                                                        datadir=datadir,
                                                        bidsdir=bidsdir,
                                                        condition='left-right',
                                                        timespan=timespan_right)
    # define trialorder. according to reward magnitude below
    trialorder = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    # order according to probability:
    # trialorder = ['E', 'H', 'I', 'C', 'F', 'A', 'G', 'B', 'D']
    # order according to expected value (prob*reward):
    # trialorder = ['A', 'C', 'E', 'B', 'F', 'H', 'D', 'G', 'I']
    results = _create_splits_from_left_and_right_stimulation(
        subjects=subjects,
        ntrain=ntrain,
        ntest=ntest,
        trialorder=trialorder,
        leftdata=leftdata,
        rightdata=rightdata)
    # create plots based on the data
    models = plot_many_distance_matrices(results=results,
                                         triallength=triallength,
                                         figdir=figdir,
                                         subjects=subjects)
    return results, models


def plot_many_distance_matrices(results,
                                triallength,
                                figdir,
                                subjects):
    """
    Plot a variety of distance matrices from raw, model, and transformed data
    on single-subject and group level.
    """
    # create a timestamp and include it in the plot title
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # create & plot distance matrices of SRMs with different no of components,
    # fit on the averaged and unaveraged artificial train timeseries. Return SRM
    # loop through different number of features
    models = {}
    for n in [5, 10, 20, 40, 80, 160]:
        models[n] = {}
        # This fits a probabilistic SRM with n features and returns the model
        # Based on the shared response space of the model, it plots the
        # correlation distance between all combinations of trial types in the
        # data (here: left and right visual stimulation, averaged)
        # RELEVANCE: This plot shows whether trial information from the
        # experiment structure is preserved in the model build from averaged
        # trials
        title = f'Correlation distance of trialtypes ({triallength*10}ms) \n' \
                f'in shared space ({n} components), fit on left and right \n' \
                f'averaged training data. Created: {timestr}'
        name = f'corr-dist_avg-traindata_full-stim_{n}-components.png'
        models[n]['full'] = plot_trialtype_distance_matrix(
            results['averaged_trials']['train']['full'],
            n,
            figdir=figdir,
            triallength=triallength,
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name
            )
        # This fits a probabilistic SRM with n features and returns the model
        # Based on the shared response space of the model, it plots the
        # correlation distance between all combinations of trial types in the
        # data (here: left visual stimulation only, averaged)
        # Importantly the plot is scaled (clim) to enhance correlation patterns
        # RELEVANCE: This plot shows whether trial information from the
        # experiment structure is preserved in the model build from averaged
        # trials
        title = f'Correlation distance of trialtypes ({triallength*10}ms) \n' \
                f'in shared space ({n} components), fit on left  \n' \
                f'averaged training data. Created: {timestr}'
        name = f'corr-dist_avg-traindata_left-stim_{n}-components.png'
        models[n]['left'] = plot_trialtype_distance_matrix(
            results['averaged_trials']['train']['left'],
            n,
            figdir=figdir,
            trialtypes=9,
            clim=[0, 0.5],
            triallength=triallength,
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name
        )
        # This fits a probabilistic SRM with n features and returns the model
        # Based on the shared response space of the model, it plots the
        # correlation distance between all combinations of trial types in the
        # data (here: left and right visual stimulation, original time series
        # (not-averaged!))
        # RELEVANCE: This plot shows whether trial information from the
        # experiment structure is preserved in the model build from individual
        # trials
        title = f'Correlation distance of trialtypes ({triallength*10}ms) \n' \
                f'in shared space ({n} components), fit on left and right \n' \
                f'original training data. Created: {timestr}'
        name = f'corr-dist_orig-traindata_full-stim_{n}-components.png'
        plot_trialtype_distance_matrix(
            results['original_trials']['train']['full'],
            n,
            figdir=figdir,
            trialtypes=270,
            triallength=triallength,
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name
        )

    # create subject specific and averaged distance matrices from raw data

    # This plot uses the MEG data and computes correlation distances between the
    # data in each trial type (here, on averaged train data). It is the baseline
    # of which trialtype-experiment structure is embedded into the data that the
    # model is trained on. It does so subject-wise, but it also averages the
    # subject-wise distance matrices
    # RELEVANCE: TODO
    compute_raw_distances(
        results['averaged_trials']['train']['full'],
        subjects,
        figdir=figdir,
        trialtypes=18,
        triallength=triallength,
        feature='avg-train',
        nametmpl='group_raw_avg-train_trialdist_18.png',
        y_label='trialtype',
        x_label='trialtype',
        timestr=timestr
        )
    # This plot uses the MEG data and computes correlation distances between the
    # data in each trial type (here, on averaged test data). It is the baseline
    # of which trialtype-experiment structure is embedded into the data that the
    # model has not seen yet It does so subject-wise, but it also averages the
    # subject-wise distance matrices
    # RELEVANCE: TODO
    compute_raw_distances(
        results['averaged_trials']['test']['full'],
        subjects,
        figdir=figdir,
        trialtypes=18,
        triallength=triallength,
        feature='avg-test',
        nametmpl='group_raw_avg-test_trialdist_18.png',
        y_label='trialtype',
        x_label='trialtype',
        timestr=timestr
        )

    # transform subject raw data into shared model room, plot subject specific
    # and averaged distance matrices

    # This plot takes subject data in shared response space that the model has
    # not seen during training, and creates a distance matrix between the data
    # in all trialtypes for each subject.
    # It is an indicator whether trialtype-experiment
    # features are present in data transformed with the model. It can be
    # compared to the distance matrices that were created from non-transformed
    # MEG data of the same subject. It also averages all individual subject
    # distance matrices into one matrix
    for n in [5, 10, 20, 40, 80, 160]:
        shared_test = models[n]['full'].transform(
            results['averaged_trials']['test']['full'])
        compute_raw_distances(
            data=shared_test,
            subjects=subjects,
            figdir=figdir,
            trialtypes=18,
            triallength=triallength,
            feature='transformed-avg-test',
            nametmpl=f'group_srm-{n}_transformed-avg-test_trialdist-18.png',
            y_label='trialtype',
            x_label='trialtype',
            timestr=timestr,
            )
        # average the transformed time series across subjects, build a single
        # distance matrix from this
        mean_shared_test = np.mean(shared_test, axis=0)
        assert mean_shared_test.shape[0] == n
        # make the distance matrix
        # this plot displays correlation distance between data from different
        # trials in the experiment, computed from unseen test data that was
        # transformed into the shared space.
        title = f"Trialtype-by-trialtype distance between averaged \n" \
                f"transformed ({n} components) test data."
        name = \
            f'group_avg-transformed_transformed-n-{n}-avg-test_trialdist-18.png'
        mean_shared_test_dist = plot_trialtype_distance_matrix(
            mean_shared_test,
            n='group',
            figdir=figdir,
            trialtypes=18,
            on_model_data=False,
            triallength=triallength,
            feature=f'transformed-test-average-with-srm{n}',
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name
        )
        logging.info(f'Permutation test on transformed test data with {n} '
                     f'components:')
        correlate_distance_matrix_quadrants(
            mean_shared_test_dist,
            figdir=figdir + '/group/meg',
            name=f'averaged_transformed_test_data_srm{n}.png')
        # This plot takes subject data in shared response space that the model
        # was trained on, and creates a distance matrix between the data
        # in all trialtypes for each subject.
        # It is an indicator whether trialtype-experiment
        # features are present in data transformed with the model. It can be
        # compared to the dist matrices that were created from non-transformed
        # MEG data of the same subject. It also averages all individual subject
        # distance matrices into one matrix
        shared_train = models[n]['full'].transform(
            results['averaged_trials']['train']['full'])
        compute_raw_distances(
            data=shared_train,
            subjects=subjects,
            figdir=figdir,
            trialtypes=18,
            triallength=triallength,
            feature='transformed-avg-train',
            nametmpl=f'group_srm-{n}_transformed-avg-train_trialdist-18.png',
            y_label='trialtype',
            x_label='trialtype',
            timestr=timestr,
            )
        # average the transformed time series across subjects, build a single
        # distance matrix from this
        mean_shared_train = np.mean(shared_train, axis=0)
        assert mean_shared_train.shape[0] == n
        # make the distance matrix
        title = f"Trialtype-by-trialtype distance between averaged \n" \
                f"transformed ({n} components) train data."
        name = \
            f'group_avg-transformed_transformed-n-{n}-avg-train_trialdist-18.png'
        mean_shared_train_dist = plot_trialtype_distance_matrix(
            mean_shared_train,
            n='group',
            figdir=figdir,
            trialtypes=18,
            on_model_data=False,
            triallength=triallength,
            feature=f'transformed-train-average-with-srm{n}',
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name
        )
        logging.info(f'Permutation test on transformed train data with {n} '
                     f'components:')
        correlate_distance_matrix_quadrants(
            mean_shared_test_dist,
            figdir=figdir + '/group/meg',
            name=f'averaged_transformed_train_data_srm{n}.png'
        )
    # finally, average non-transformed timeseries over subjects,
    # and build distance matrices
    for (data, label) in \
            [(results['averaged_trials']['train']['full'], 'avg-raw-train'),
             (results['averaged_trials']['test']['full'], 'avg-raw-test')]:
        mean = np.mean(data, axis=0)
        assert mean.shape[0] == 306
        title = f'Trialtype-by-trialtype correlation distance on \n' \
                f'{label} data. Created: {timestr}'
        name = f'groupavg_{label}_trialdist-18.png'
        # make the distance matrix
        plot_trialtype_distance_matrix(mean,
                                       n='group',
                                       figdir=figdir,
                                       trialtypes=18,
                                       on_model_data=False,
                                       triallength=triallength,
                                       feature=label,
                                       y_label='trialtype',
                                       x_label='trialtype',
                                       title=title,
                                       name=name
                                       )

    return models


def correlate_distance_matrix_quadrants(distmat, figdir, name):
    """
    Take the lower triangle matrix of each quadrant in the distance matrix,
    and then correlate all quadrants lower triangle matrices with eachother.
    Repeat this process, but with permutated values.
    Find the percentile of the non-permutated value within the permutated ones.
    :param distmat: array, contains the values of the distance matrix
    :param figdir: str, path to where figures are saved
    :param name: str, file name
    :return:
    """
    # ensure that the distance matrix has the expected shape
    assert distmat.shape == (18, 18)
    # split it into quadrants:
    #  |a|c|
    #  |b|d|
    #
    a, b, c, d = distmat[:9, :9], distmat[9:, :9], \
                 distmat[:9, 9:], distmat[9:, 9:]
    # shift the lower triangle one to the left, to not include the diagonal
    inds = np.tril_indices(9, -1)
    # transform all quadrants lower triangles into arrays
    a_1d = np.asarray(a[inds])
    b_1d = np.asarray(b[inds])
    c_1d = np.asarray(c[inds])
    d_1d = np.asarray(d[inds])
    observed_correlations = _get_quadrant_corrs(a_1d, b_1d, c_1d, d_1d)
    # now, permutate the inner structure of the vectors and recompute the
    # correlation 10000 times
    rng = np.random.default_rng()
    permutations = []
    for i in range(10000):
        for q in [a_1d, b_1d, c_1d, d_1d]:
            # TODO: potentially wrong, ask mih for dist matrix
            # do a mantel test
            rng.shuffle(q)
        permutations.append(_get_quadrant_corrs(a_1d, b_1d, c_1d, d_1d))

    results = {}
    from scipy.stats import percentileofscore
    import seaborn as sns
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}
    plt.rc('font', **font)
    for quad_combination, idx in [('ad', 0), ('ab', 1), ('ac', 2), ('db', 3),
                                  ('dc', 4)]:
        dist = [l[idx] for l in permutations]
        perc = percentileofscore(dist, observed_correlations[idx])
        results[quad_combination] = perc
        # plot a histogram
        fig = sns.histplot(dist, kde=True)
        plt.axvline(observed_correlations[idx], c='black', ls='--', lw=1)
        plt.title(f'Distribution of random correlations between quadrants \n'
                  f'{quad_combination[0]} and {quad_combination[1]}, '
                  f'and observed correlation')
        plt.text(observed_correlations[idx]-0.25, 300,
                 f'r={observed_correlations[idx]:.2f}, \n'
                 f'({perc}th)')
        plt.text(-0.6, 450, " a  c \n b  d ", fontweight='light',
                 bbox={'boxstyle': 'square', 'facecolor': 'white'})
        plt.xlabel('Pearson correlation')
        fname = figdir + '/' + f'correlation_{quad_combination}_{name}'
        plt.savefig(fname)
        plt.close()
        logging.info(f'Saving results of the permutation test to {fname}')
    print(results)

    return results


def _get_quadrant_corrs(a, b, c, d):
    """
    get the correlations between four quadrants
    :param a, b, c, d: array, 1D vector corresponding to the lower triangle of
     each quadrant of a distance matrix
    :return:
    """
    # correlation between the intertrial distance of the left set of stimuli
    # and the intertrial distance of the right stimuli
    ad = np.corrcoef(a, d)[0][1]
    ab = np.corrcoef(a, b)[0][1]
    ac = np.corrcoef(a, c)[0][1]
    db = np.corrcoef(d, b)[0][1]
    dc = np.corrcoef(d, c)[0][1]
    return ad, ab, ac, db, dc



def compute_raw_distances(data,
                          subjects,
                          figdir,
                          trialtypes,
                          triallength,
                          feature=None,
                          nametmpl='group_task-memento_raw_avg.png',
                          y_label='trialtype',
                          x_label='trialtype',
                          subjecttitle = None,
                          grouptitle = None,
                          timestr=None,
                          ):
    """
    Take a list of lists with time series from N subjects.
    For each subject/list in this data, build a trialtype-by-trialtype
    correlation distance matrix.

    Then, take the N distance matrices, transform them from correlation distance
    to correlation, and Fisher-z-transform them. Afterwards, average them across
    subjects
    :param data: Data to use for distance matrix creation. Should be a list of
    arrays
    :param subjects: list, of subject identifiers
    :param feature: str, some additional identifier, used in figure name
    :param figdir: str, path to where plots are saved
    :param triallength: int, length of a single trial in samples
    :param trialtypes: Number of consecutive trial types in the time series to
    plot. 9 or 18 for single/averaged trials left or left+right, or n*trialtypes
    :return:
    """
    # create & plot distance matrices of raw data per subject. Return dist_mat
    distmat = {}
    prev_subjecttitle = None
    for idx, sub in enumerate(subjects):
        if subjecttitle is None or subjecttitle == prev_subjecttitle:
            subjecttitle = f'Correlation distance of trialtypes ' \
                           f'({triallength*10}ms) \n in {feature} data of' \
                           f' subject {sub}. Created: {timestr}'
            # the following is necessary to update the subject id in the title
            prev_subjecttitle = subjecttitle
        type = 'full-stim' if trialtypes in [18, 270] else 'left-stim'
        name = f'sub-{sub}_corr-dist_{feature}data_{type}.png'
        distmat[sub] = \
            plot_trialtype_distance_matrix(data[idx],
                                           n=sub,
                                           figdir=figdir,
                                           trialtypes=trialtypes,
                                           on_model_data=False,
                                           triallength=triallength,
                                           feature=feature,
                                           y_label=y_label,
                                           x_label=x_label,
                                           title=subjecttitle,
                                           name=name
                                           )
    # Fisher-z transform the matrices
    zdistmat = {}
    for sub in subjects:
        # transform data from correlation distance back to correlation
        corrdist = 1 - distmat[sub]
        assert (corrdist >= -1).all() & (corrdist <= 1).all(),\
            "We have impossible correlations"
        # fisher z-transform
        print(sub)
        zdistmat[sub] = np.arctanh(corrdist)
    # average the matrices
    avg = np.mean(np.array([v for k, v in zdistmat.items()]), axis=0)
    # transform it back to correlation distance
    # TODO: backcheck wither this is correct!
    avg = np.nan_to_num(avg, posinf=1)
    avg_corrdist = 1 - avg
    assert not (avg_corrdist <= -1).any()
    assert not (avg_corrdist >= 1).any()
    # plot it
    # set font specifications for plots
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 50}
    plt.rc('font', **font)
    plt.figure(figsize=[50, 50])
    plt.imshow(avg_corrdist, cmap='viridis')
    plt.colorbar()
    # set a figure title according to the number of trialtypes plotted
    type = 'left and right' if trialtypes in [18, 270] else 'left'
    if grouptitle is None:
        grouptitle = f"Average of subject-wise {feature} data trial \n" \
                     f"distances for {type} stimulation. Created: {timestr}"
    plt.title(grouptitle)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fname = Path(figdir) / f'group/meg' / f'{nametmpl}'
    logging.info(f'Saving averaged distance matrix to {fname}')
    plt.savefig(fname)


def plot_trialtype_distance_matrix(data,
                                   n,
                                   figdir,
                                   trialtypes=18,
                                   clim=None,
                                   on_model_data=True,
                                   feature=None,
                                   triallength=70,
                                   title=None,
                                   y_label=None,
                                   x_label=None,
                                   name=None):
    """
    A generic function to fit SRMs and plot distance matrices on trial data.
    In hopes of getting accurate plot titles and names, I'm heavily
    parameterizing the function to build or include custom specifications.
    :param data: list of lists, needs to be one list per subject with time
    series data. SRM is fit on this data
    :param n: Number of features to use in SRM
    :param figdir: Path to where figures and plots are saved
    :param trialtypes: Number of trialtypes to plot. 9 if only left trials are
    used, 18 when its both left and right, 135 when its left but not averaged
    :param clim: list, upper and lower limit of the colorbar
    :param feature: str, some additional identifier, used in figure name
    :param on_model_data: bool, if True, fit an SRM model and base the distance
    matrix on the shared components. If False, base the distance matrix on raw
    data. In the latter case, data needs to be a single time series, no list of
    lists, and n needs to be a subject identifier
    :param triallength: int, length of a single trial in samples. Defaults to 70
    :return:
    """
    if clim is None:
        # default to a range of 0 to 1
        clim = [0, 1]
    if on_model_data:
        assert len(data) > 1
        assert isinstance(data, list)
        # fit a probabilistic SRM
        model = shared_response(data, features=n)
        # get the componentXtime series of each trial in SRM, and put it into
        # a nested array. 70 -> length of one trial / averaged trial type
        trialmodels_ = np.array(
            [model.s_[:, triallength * i:triallength * (i + 1)].ravel()
             for i in range(trialtypes)])
    else:
        assert isinstance(data, np.ndarray)
        trialmodels_ = np.array(
            [data[:, triallength * i:triallength * (i + 1)].ravel()
             for i in range(trialtypes)])

    dist_mat = sp_distance.squareform(
        sp_distance.pdist(trialmodels_, metric='correlation'))
    # set font specifications for plots
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 50}
    plt.rc('font', **font)
    plt.figure(figsize=[50, 50])
    plt.imshow(dist_mat, cmap='viridis', clim=clim)

    if title is None:
        # set a figure title according to the number of trialtypes plotted
        type = 'left and right' if trialtypes in [18, 270] else 'left'
        if on_model_data:
            title = f"Correlation distance between trialtypes for \n" \
                    f"{type} stimulation"
        else:
            title = f"Correlation distance between trialtypes for \n" \
                    f"{type} stimulation in raw data"
    title = title
    plt.title(title)
    if x_label is None:
        x_label = 'trial type'
    if y_label is None:
        y_label = 'trial type'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if name is None:
        datatype = 'srm' if on_model_data else 'raw'
        name = f'trialtype-distance_{datatype}-data_{trialtypes}-trials_n-{n}_feature-{feature}.png'
    fname = Path(figdir) / f'group/meg' / name
    plt.colorbar()
    logging.info(f"Saving a distance matrix to {fname}")
    plt.savefig(fname)
    plt.close('all')
    if on_model_data:
        return model
    else:
        return dist_mat


def concatenate_data(data, field='normalized_data'):
    """
    Concatenate trial data in a list of dictionaries
    :param data: nested dict, contains all trial infos
    :param field: str, dict key in info dict in general data structure
    :return:
    """
    time_series = np.concatenate([info[field] for info in data],
                                 axis=1)

    assert time_series.shape[0] == 306
    return time_series


def concatenate_means(data):
    time_series = np.concatenate(data, axis=1)
    assert time_series.shape[0] == 306
    return time_series


def plot_trial_components_from_srm(subject,
                                   datadir,
                                   bidsdir,
                                   figdir,
                                   condition='left-right',
                                   timespan='fulltrial'):
    """
    Previously, we fit a probabilistic SRM to data, transformed the data
    with each trial's  weights. Now, we plot the data feature-wise for
    different conditions.
    :param subject
    :param datadir
    :param bidsdir
    :param figdir
    :param condition: str, an identifier based on which trials can be split.
    Possible values: 'left-right' (left versus right option choice),
    'nobrain-brain' (no-brainer versus brainer trials)
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), 'fulltrial' (entire 7second epoch), 'secondstim',
    'delay'.
    :return:
    models, dict with feature number as keys and corresponding models as values
    data: list of dicts, with all trial information
    """
    fullsample, data = get_general_data_structure(subject=subject,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition=condition,
                                                  timespan=timespan)

    logging.info(f'Fitting shared response models based on data from subjects '
                 f'{subject}')
    # use a wide range of features to see if anything makes a visible difference
    features = [5, 10, 20, 100]
    models = {}
    for f in features:
        # fit the model. List comprehension to retrieve the data as a list of
        # lists, as required by brainiak
        model = shared_response(data=[d['normalized_data'] for d in data],
                                features=f)
        final_df, data = create_full_dataframe(model, data)

        # plot individual features
        plot_srm_model(df=final_df,
                       nfeatures=f,
                       figdir=figdir,
                       subject='group',
                       mdl='srm',
                       cond=condition,
                       timespan=timespan)
        models[f] = model
    for idx, model in models.items():
        plot_distance_matrix(model, idx, figdir)
    return models, data


def add_trial_types(subject,
                    bidsdir,
                    all_trial_info):
    """
    Bring trials in a standardized sequence across participants according to
    their characteristics

    Left characteristics
    Event name ->   description ->      typename:   -> count
    lOpt1 -> LoptMag 0.5, LoptProb 0.4 -> A ->  70
    lOpt2 -> LoptMag 0.5, LoptProb 0.8 -> B ->  65
    lOpt3 -> LoptMag 1, LoptProb 0.2 -> C ->    50
    lOpt4 -> LoptMag 1, LoptProb 0.8 -> D ->    70
    lOpt5 -> LoptMag 2, LoptProb 0.1 -> E ->    50
    lOpt6 -> LoptMag 2, LoptProb 0.2 -> F ->    35
    lOpt7 -> LoptMag 2, LoptProb 0.4 -> G ->    50
    lOpt8 -> LoptMag 4, LoptProb 0.1 -> H   ->  70
    lOpt9 -> LoptMag 4, LoptProb 0.2 -> I   ->  50

    Right_characteristics:

    :return:
    """
    stim_char = get_stimulus_characteristics(subject,
                                             bidsdir,
                                             columns=['trial_no',
                                                      'LoptMag',
                                                      'LoptProb',
                                                      'RoptMag',
                                                      'RoptProb',
                                                      'RT',
                                                      'choice',
                                                      'pointdiff']
                                             )

    # add the probability and magnitude information
    for info in all_trial_info.keys():
        LMag = stim_char[stim_char['trial_no'] == info]['LoptMag'].item()
        LPrb = stim_char[stim_char['trial_no'] == info]['LoptProb'].item()
        RMag = stim_char[stim_char['trial_no'] == info]['RoptMag'].item()
        RPrb = stim_char[stim_char['trial_no'] == info]['RoptProb'].item()
        RT = stim_char[stim_char['trial_no'] == info]['RT'].item()
        choice = stim_char[stim_char['trial_no'] == info]['choice'].item()
        pointdiff = stim_char[stim_char['trial_no'] == info]['pointdiff'].item()
        # also add information what the previous trial was, if we have info
        if info > 1:
            prev_choice = stim_char[stim_char['trial_no'] == info - 1]['choice'].item()
            prev_RT = stim_char[stim_char['trial_no'] == info - 1]['RT'].item()
            prev_LMag = stim_char[stim_char['trial_no'] == info - 1]['LoptMag'].item()
            prev_LPrb = stim_char[stim_char['trial_no'] == info - 1]['LoptProb'].item()
            prev_RMag = stim_char[stim_char['trial_no'] == info - 1]['RoptMag'].item()
            prev_RPrb = stim_char[stim_char['trial_no'] == info - 1]['RoptProb'].item()
        else:
            # this should be ok because None is a singleton
            prev_choice = prev_RT = prev_RPrb = prev_RMag = prev_LPrb = prev_LMag = None
        all_trial_info[info]['prevLoptMag'] = prev_LMag
        all_trial_info[info]['prevLoptProb'] = prev_LPrb
        all_trial_info[info]['prevRoptMag'] = prev_RMag
        all_trial_info[info]['prevRoptProb'] = prev_RPrb
        all_trial_info[info]['prevLchar'] = trial_characteristics[
            (prev_LMag, prev_LPrb)] if prev_LMag else None
        all_trial_info[info]['prevRchar'] = trial_characteristics[
            (prev_RMag, prev_RPrb)] if prev_RMag else None
        all_trial_info[info]['prevRT'] = prev_RT
        all_trial_info[info]['prevchoice'] = prev_choice
        all_trial_info[info]['LoptMag'] = LMag
        all_trial_info[info]['LoptProb'] = LPrb
        all_trial_info[info]['RoptMag'] = RMag
        all_trial_info[info]['RoptProb'] = RPrb
        all_trial_info[info]['Lchar'] = trial_characteristics[(LMag, LPrb)]
        all_trial_info[info]['Rchar'] = trial_characteristics[(RMag, RPrb)]
        all_trial_info[info]['RT'] = RT
        all_trial_info[info]['choice'] = choice
        all_trial_info[info]['pointdiff'] = pointdiff

    # get a count of trials per characteristic
    Lchars = [info['Lchar'] for info in all_trial_info.values()]
    Rchars = [info['Rchar'] for info in all_trial_info.values()]
    from collections import Counter
    Lcounts = Counter(Lchars)
    Rcounts = Counter(Rchars)
    return all_trial_info


def plot_distance_matrix(model, idx, figdir):
    """
    plot a distance matrix between time points from the shared response.
    :param model:
    :param idx
    :param figdir:
    :return:
    """
    dist_mat = sp_distance.squareform(sp_distance.pdist(model.s_.T,
                                                        metric='correlation'))
    plt.xlabel('t (100 = 1sec)')
    plt.ylabel('t (100 = 1sec)')
    plt.imshow(dist_mat, cmap='viridis')
    # TODO: maybe add vertical lines in experiment landmarks
    plt.colorbar()
    fname = Path(figdir) / f'group/meg' / \
                            f'group_task-memento_srm-{idx}_distances.png'
    plt.savefig(fname)
    plt.close()


def create_full_dataframe(model,
                          data):
    """
    Create a monstrous pandas dataframe.

    :param model: Brainiak SRM model, fitted
    :param data: List of Pandas dataframes with MEG data and trial info
    :return: data: completed List of pd DataFrames with transformed data
    :return: finaldf: Large pd Dataframe, used for plotting
    """
    # transform the data with the fitted model. list comprehension as data is
    # a list of dicts
    transformed = model.transform([d['normalized_data'] for d in data])
    # add the transformed data into the dicts. They hold all information
    for d in data:
        d['transformed'] = transformed.pop(0)
    assert transformed == []
    # build a dataframe for plotting
    dfs = []
    for d in data:
        df = pd.DataFrame.from_records(d['transformed']).T
        trial_type = d['trial_type']
        trial = d['trial_no']
        df['trial_type'] = trial_type
        df['trial_no'] = trial
        dfs.append(df)

    finaldf = pd.concat(dfs)
    return finaldf, data


def combine_data(df,
                 sub,
                 trials_to_trialtypes,
                 epochs_to_trials,
                 bidsdir,
                 timespan,
                 freq=100):
    """
    Generate a dictionary that contains all relevant information of a given
    trial, including the data, correctly slices, to train the model on.
    :param df: pandas dataframe, contains the MEG data
    :param sub: str; subject identifier
    :param trials_to_trialtypes: Dict; a mapping of trial numbers to trial type
    :param epochs_to_trials: Dict; a mapping of epochs to trial numbers
    :param bidsdir; str, Path to BIDS dir with log files
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), or 'fulltrial' (entire 7second epoch), 'secondstim',
    'delay', or a time frame within the experiment in samples (beware of the
    sampling rate!)
    :param freq: int, frequency of the data. Earlier versions of this code
    required 100Hz sampling rate, this parameter tries to make this more
    flexible
    :return: all_trial_infos; dict; with trial-wise information
    """
    all_trial_infos = {}
    unique_epochs = df['epoch'].unique()

    if timespan == 'decision':
        # extract the information on decision time for all trials at once.
        trials_to_rts = get_decision_timespan_on_and_offsets(subject=sub,
                                                             bidsdir=bidsdir)
    if isinstance(timespan, list):
        logging.info(f"Received a list as a time span. Attempting to "
                     f"subset the available trial data in range "
                     f"{timespan[0]}, {timespan[1]} samples")
    else:
        logging.info(f"Selecting data for the event description {timespan}.")

    for epoch in unique_epochs:
        # get the trial number as a key
        trial_no = epochs_to_trials[epoch]
        # get the trial type, if it hasn't been excluded already
        trial_type = trials_to_trialtypes.get(trial_no, None)
        if trial_type is None:
            continue
        # get the data of the epoch. Transpose it, to get a sensors X time array
        data = df.loc[df['epoch'] == epoch, 'MEG0111':'MEG2643'].values.T
        if timespan == 'fulltrial':
            # the data does not need to be shortened. just make sure it has the
            # expected dimensions
            assert data.shape == (306, 7*freq)
        elif timespan == 'firststim':
            # we only need the first 700 milliseconds from the trial,
            # corresponding to the first 70 entries since we timelocked to the
            # onset of the first stimulation
            data = data[:, :0.7*freq]
            assert data.shape == (306, 0.7*freq)
        elif timespan == 'delay':
            # take the 2 seconds after the first stimulus
            data = data[:, (0.7*freq):(2.7*freq)]
            assert data.shape == (306, 2*freq)
        elif timespan == 'secondstim':
            # take 700 ms after the first stimulus + delay phase
            data = data[:, (2.7*freq):(3.4*freq)]
            assert data.shape == (306, 0.7*freq)
        elif timespan == 'decision':
            # we need an adaptive slice of data (centered around the exact time
            # point at which a decision was made in a given trial.
            if trial_no not in trials_to_rts.keys():
                # if the trial number has been sorted out before, don't append
                # the data
                continue
            onset, offset = trials_to_rts[trial_no]
            data = data[:, onset:offset]
            assert data.shape == (306, 0.8*freq)
        else:
            if isinstance(timespan, list):
                data = data[:, timespan[0]:timespan[1]]
            else:
                raise NotImplementedError(f"The timespan {timespan} is not "
                                          f"implemented.")
        # normalize (z-score) the data within sensors
        normalized_data = stats.zscore(data, axis=1, ddof=0)
        all_trial_infos[trial_no] = {'epoch': epoch,
                                     'trial_type': trial_type,
                                     'data': data,
                                     'normalized_data': normalized_data}
    return all_trial_infos


def get_decision_timespan_on_and_offsets(subject,
                                         bidsdir,
                                         freq=100):
    """
    For each trial, get a time frame around the point in time that a decision
    was made.
    :param subject; str, subject identifier
    :param bidsdir: str, path to BIDS dir with log files
    :return: trials_to_rts: dict, and association of trial numbers to 800ms time
    slices around the time of decision in the given trial
    """
    logs = read_bids_logfile(subject=subject,
                             bidsdir=bidsdir)
    trials_and_rts = logs[['trial_no', 'RT']].values
    logging.info(f'The average reaction time was '
                 f'{np.nanmean(trials_and_rts[:,1])}')
    # mark nans with larger RTs
    logging.info('Setting nan reaction times to implausibly large values.')
    np.nan_to_num(trials_and_rts, nan=100, copy=False)
    # collect the trial numbers where reaction times are too large to fit into
    # the trial. For now, this is at 3 seconds.
    trials_to_remove = trials_and_rts[np.where(
        trials_and_rts[:, 1] > 3)][:, 0]
    # initialize a dict to hold all information
    trials_to_rts = {}
    for trial, rt in trials_and_rts:
        # get the right time frame. First, transform decision time into the
        # timing within the trial. this REQUIRES a sampling rate of 100Hz!
        rt = rt * freq
        # Now, add RT to the start of the second visual stimulus to get the
        # approximate decision time from trial onset
        # (0.7s + 2.0s = 2.7s)
        decision_time = rt + 2.7 * freq
        # plausibility check, no decision is made before a decision is possible
        assert decision_time > 2.7 * freq
        # calculate the slice needed for indexing the data for the specific
        # trial. We round down so that the specific upper or lower time point
        # can be used as an index to subset the data frame
        slices = [int(np.floor(decision_time - (0.4 * freq))),
                  int(np.floor(decision_time + (0.4 * freq)))]
        assert slices[1] - slices[0] == 0.8 * freq
        if trial not in trials_to_remove:
            trials_to_rts[trial] = slices

    return trials_to_rts


def shared_response(data,
                    features):
    """
    Compute a shared response model from a list of trials
    :param data: list of lists, with MEG data
    :param features: int, specification of feature number for the model
    :return:
    """
    logging.info(f'Fitting a probabilistic SRM with {features} features...')
    # fit a probabilistic shared response model
    model = srm.SRM(features=features)
    model.fit(data)
    return model


def _find_data_of_choice(df,
                         epochs,
                         subject,
                         bidsdir,
                         condition):
    """
    Based on a condition that can be queried from the log files (e.g., right or
    left choice of stimulus), return the trial names, trial types, and epoch IDs
    :param epochs: epochs object
    :param df: pandas dataframe of epochs
    :param subject: str, subject identifier '011'
    :param bidsdir: str, path to bids data with logfiles
    :param condition: str, a condition description. Must be one of 'left-right'
    (for trials with right or left choice), 'nobrain-brain' (for trials with
    no-brainer decisions versus actual decisions)
    :return: trials_to_trialtypes; dictionary with trial - condition
    :return: epochs_to_trials; dict; epoch ID to trial number associations
    """
    # get an association of trial types with trial numbers
    if condition == 'left-right':
        logging.info('Attempting to retrieve trial information for left and '
                     'right stimulus choices')
        choices = get_leftright_trials(subject=subject,
                                       bidsdir=bidsdir)
    elif condition == 'nobrain-brain':
        logging.info('Attempting to retrieve trial information for no-brainer '
                     'and brainer trials')
        choices = get_nobrainer_trials(subject=subject,
                                       bidsdir=bidsdir)

    # Create a mapping between epoch labels and trial numbers based on metadata.
    assert len(df['epoch'].unique()) == len(epochs.metadata.trial_no.values)
    # generate a mapping between trial numbers and epoch names in the dataframe
    # TODO: take this straight from metadata, epochs are indices
    epochs_to_trials = {key: value for (key, value) in
                        zip(df['epoch'].unique(),
                            epochs.metadata.trial_no.values)}
    # make sure we caught all epochs
    assert all([i in df['epoch'].unique() for i in epochs_to_trials.keys()])
    assert len(df['epoch'].unique()) == len(epochs_to_trials.keys())

    # transform the "trial_no"-"trial_type" association in choices into an
    # association of "trial_no that exist in the data" (e.g., survived
    # cleaning) - "trial_types". The epochs_to_trials mapping is used as an
    # indication with trial numbers are actually still existing in the epochs
    trials_to_trialtypes = {}
    for cond, trials in choices.items():
        # trials is a list of all trial numbers in a given condition
        counter = 0
        for trial in trials:
            # we need extra conditions because during the generation of each of
            # the dicts below, some trials may have been excluded
            if trial in epochs_to_trials.values():
                trials_to_trialtypes[trial] = cond
                counter += 1
            else:
                logging.info(f'Trial number {trial} is not '
                             f'included in the data.')

        logging.info(f"Here's my count of matching events in the SRM data for"
                     f" condition {cond}: {counter}")

    return trials_to_trialtypes, epochs_to_trials


def get_nobrainer_trials(subject, bidsdir):
    """
    Return the trials where a decision is a "nobrainer", a trial where both the
    reward probability and magnitude of one option is higher than that of the
    other option.
    :param subject: str, subject identifier
    :param bidsdir: str, path to BIDS dir with logfiles
    :return:
    """
    df = read_bids_logfile(subject=subject,
                           bidsdir=bidsdir)
    # where is the both Magnitude and Probability of reward greater for one
    # option over the other? -> nobrainer trials
    right = df['trial_no'][(df.RoptMag > df.LoptMag) &
                           (df.RoptProb > df.LoptProb)].values
    left = df['trial_no'][(df.LoptMag > df.RoptMag) &
                          (df.LoptProb > df.RoptProb)].values
    # the remaining trials require information integration ('brainer' trials)
    brainer = [i for i in df['trial_no'].values
               if i not in right and i not in left]

    # brainer and no-brainer trials should add up
    assert len(brainer) + len(right) + len(left) == len(df['trial_no'].values)
    # make sure that right and left no brainers do not intersect - if they have
    # common values, something went wrong
    assert not bool(set(right) & set(left))
    # make sure to only take those no-brainer trials where participants actually
    # behaved as expected. Those trials where it would be a nobrainer to pick X,
    # but the subject chose Y, are excluded with this.
    consistent_nobrainer_left = \
        [trial for trial in left if
         df['choice'][df['trial_no'] == trial].values == 1]
    consistent_nobrainer_right = \
        [trial for trial in right if
         df['choice'][df['trial_no'] == trial].values == 2]
    logging.info(f"Subject {subject} underwent a total of {len(right)} "
                 f"no-brainer trials for right choices, and a total of "
                 f"{len(left)} no-brainer trials for left choices. The subject "
                 f"chose consistently according to the no-brainer nature of "
                 f"the trial in N={len(consistent_nobrainer_right)} cases for "
                 f"right no-brainers, and in "
                 f"N={len(consistent_nobrainer_left)} cases for left "
                 f"no-brainers.")
    # create a dictionary with brainer and no-brainer trials. We leave out all
    # no-brainer trials where the participant hasn't responded in accordance to
    # the no-brain nature of the trial
    choices = {'brainer': brainer,
               'nobrainer_left': consistent_nobrainer_left,
               'nobrainer_right': consistent_nobrainer_right}

    return choices


def get_leftright_trials(subject,
                         bidsdir):
    """
    Return the trials where a left choice and where a right choice was made.
    Logdata coding for left and right is probably 1 = left, 2 = right (based on
    experiment file)
    :param subject: str, subject identifier, e.g., '001'
    :param bidsdir: str, Path to the root of a BIDS raw dataset
    :return: choices; dict of trial numbers belonging to left or right choices
    """
    df = read_bids_logfile(subject=subject,
                           bidsdir=bidsdir)
    # get a list of epochs in which the participants pressed left and right
    left_choice = df['trial_no'][df['choice'] == 1].values
    right_choice = df['trial_no'][df['choice'] == 2].values
    choices = {'left (1)': left_choice,
               'right (2)': right_choice}

    return choices


def plot_srm_model(df,
                   nfeatures,
                   figdir,
                   subject,
                   mdl='srm',
                   cond='left-right',
                   timespan='fulltrial',
                   freq=100):
    """
    Plot the features of a shared response model
    :param df: concatenated dataframe of trial data, transformed with the
    trial-specific mapping of the shared response model (returned by
    concatenate_transformations().
    :param figdir: str, path to directory to save figures in
    :param nfeatures: int, number of features in the model
    :param subject: str, subject identifier such as '011'
    :param mdl: Name of the SRM model to place in the figure name
    :param cond: Str, name of the condition plotted. Useful values are
     'left-right', 'nobrain-brain'
    :param timespan: Str, name of the timeframe used to fit the model
    :param freq: int, sampling rate in Hertz
    :return:
    """
    # TODO: This needs adjustment for trial names!
    import pylab
    import seaborn as sns
    logging.info('Plotting the data transformed with the SRM model.')
    title = 'Full trial duration' if timespan == 'fulltrial' else \
            'Duration of the first stimulus' if timespan == 'firststim' else \
            '400ms +/- decision time' if timespan == 'decision' else \
            None
    # TODO: this needs some indication of which subjects the plot is made from
    for i in range(nfeatures):
        fname = Path(figdir) / f'{subject}/meg' /\
                     f'{subject}_{mdl}_{nfeatures}-feat_task-{cond}_model-{timespan}_comp_{i}.png'
        if cond == 'left-right':
            fig = sns.lineplot(data=df[df['trial_type'] == 'right (2)'][i])
            sns.lineplot(data=df[df['trial_type'] == 'left (1)'][i]).set_title(title)
            fig.legend(title='Condition', loc='upper left',
                       labels=['left choice', 'right choice'])
        elif cond == 'nobrain-brain':
            fig = sns.lineplot(data=df[df['trial_type'] == 'brainer'][i])
            sns.lineplot(data=df[(df['trial_type'] == 'nobrainer_left') |
                                 (df['trial_type'] == 'nobrainer_right')][i]).set_title(title)
            fig.legend(title='Condition', loc='upper left',
                       labels=['brainer', 'nobrainer'])
        if timespan == 'fulltrial':
            # define the timing of significant events in the trial timecourse:
            # onset and offset of visual stimuli
            # TODO: added flexible freq later, check that this works
            events = [0, 0.7*freq, 2.7*freq, 3.4*freq]
            # plot horizontal lines to mark the end of visual stimulation
            [pylab.axvline(ev, linewidth=1, color='grey', linestyle='dashed')
             for ev in events]

        plot = fig.get_figure()
        plot.savefig(fname)
        plot.clear()


def _select_channels(epochs):
    """
    Select a subset of channels based on location in helmet
    :param epochs: pandas DataFrame, df of epochs
    :return:
    """

    right_chs = mne.read_vectorview_selection(['Right-occipital'])
    idx_right = [epochs.columns.get_loc(s.replace(' ', '')) for s in right_chs]
    left_chs = mne.read_vectorview_selection(['Left-occipital'])
    idx_left = [epochs.columns.get_loc(s.replace(' ', '')) for s in left_chs]
    idx_left = [186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                199, 200, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                224, 234, 235, 236, 237, 238, 239, 246, 247, 248]
    idx_right = [231, 232, 233, 240, 241, 242, 243, 244, 245, 261, 262, 263,
                 264, 265, 266, 267, 268, 269, 270, 271, 272, 279, 280, 281,
                 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296]
