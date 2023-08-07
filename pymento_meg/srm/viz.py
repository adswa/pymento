"""
Module for plots related to shared response models.

"""

import logging
import mne
import time
from pathlib import Path
from textwrap import wrap
from pymento_meg.utils import _construct_path
from pymento_meg.srm.utils import (
    _get_mean_and_std_from_transformed,
    _get_trial_indicators,
    _get_quadrant_corrs,
)
import scipy.spatial.distance as sp_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


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
    fname = 'TODO'
    fig.savefig(fname)


def _plot_helper(k,
                 suptitle,
                 name,
                 figdir,
                 palette='husl',
                 npalette=None,
                 figsize=(10, 2),
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
    # adjust the figsize by number of components that get plotted
    figsize = (figsize[0], figsize[1]*k)
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


# noinspection PyUnresolvedReferences
def _plot_raw_comparison(data, dataset, adderror=False, stderror=False,
                         freq=1000,
                         window=0.5, figdir='/tmp'):
    """
    Do comparison time series plots with raw data. Used to check if the raw data
    contains similar signal as the signal seen in dimensionality-reduced data.
    :param data: dict, either train_series or test_series
    :param dataset: dict, either trainset or testset
    :return:
    """
    # first, not centered, averaged over all subjects, per sensor:
    d = np.asarray([np.mean(np.asarray(i), axis=0) for i in data.values()])
    mean = np.mean(d, axis=0)
    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Average raw signal over all subjects per sensor',
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
                     suptitle='Average raw signal over all subjects & sensors',
                     name=f"avg-signal_raw-avg.png",
                     figdir=figdir,
                     npalette=1,
                     figsize=(10, 5)
                     )
    ax.plot(moremean.T, label='averaged raw data', color=palette[0])
    if adderror:
        ax.fill_between(range(len(moremean)), moremean - std, moremean + std,
                        alpha=0.4,
                        color=palette[0])
    fig.savefig(fname)
    # now response centered
    RT = [np.round(epoch['RT'] * freq)
          for subject in data for epoch in dataset[subject]]
    # time window centered around the reaction
    win = window * freq
    d = []
    [d.extend(data[i]) for i in data.keys()]
    assert len(d) == len(RT)

    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Averaged raw signal, over all subjects, '
                              'per sensor, response-locked',
                     name=f"avg-signal_raw-sensors_response-locked.png",
                     figdir=figdir,
                     npalette=1,
                     vline=win / 2,
                     figsize=(10, 5)
                     )
    # first, averaged over all subjects, per sensor
    # get all epochs
    centered_epochs = [d[idx][:, int(rt - win / 2):int(rt + win / 2)]
                       for idx, rt in enumerate(RT) if not np.isnan(rt)]
    # if an epoch does not have enough data (too short), don't use it
    d_long_enough = np.asarray(
        [e for e in centered_epochs if e.shape[1] == win])
    # average over epochs
    avg_epochs = np.mean(d_long_enough, axis=0)
    # plot
    ax.plot(avg_epochs.T, label='averaged raw data (306 sensors)')
    fig.savefig(fname)

    # average over sensors
    palette, fig, ax, fname = \
        _plot_helper(1,
                     suptitle='Averaged raw signal, over all subjects and '
                              'sensors, response-locked',
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
                     suptitle='Average raw signal, response-locked, left vs. '
                              'right',
                     name=f"avg-signal_raw-avg_response-locked_leftvsright.png",
                     figdir=figdir,
                     npalette=2,
                     vline=win / 2,
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


# noinspection PyUnresolvedReferences
def _plot_transformed_components(transformed,
                                 k,
                                 data,
                                 adderror=False,
                                 figdir='/tmp',
                                 stderror=False,
                                 modelfit=None,
                                 custom_name_component='',
                                 ):
    """
    For transformed data containing the motor response/decision, create a range
    of plots illustrating the data in shared space. The special sauce is
    centering plots around the event in question.

    :param transformed: dict, raw data transformed into shared space
    :param data: either trainset or testset
    :param k: int, number of features in the model
    :param adderror: bool, whether to add the standard deviation around means
    :param figdir: str, Path to a place to save figures
    :param stderror: bool, if true, SEM is used instead of std
    :param: custom_name_component: str, if given its added to a plot's file name
    :return:
    """
    # plot transformed components:
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Averaged signal in shared space, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per"
                          f"-comp{custom_name_component}.png",
                     figdir=figdir,
                     npalette=k,
                     )
    for i in range(k):
        mean, std = _get_mean_and_std_from_transformed(transformed, i,
                                                       stderror=stderror)
        ax[i].plot(mean, color=palette[i], label=f'component {i + 1}')
        if adderror:
            # to add std around the mean. We didn't find expected
            # congruency/reduced variability in those plots.
            ax[i].fill_between(range(len(mean)), mean - std, mean + std,
                               alpha=0.4,
                               color=palette[i])
    for a in ax:
        a.legend(loc='upper right',
                 prop={'size': 6})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname)

    if modelfit == 'trialtype':
        # the testset of this data is differently structured and the code below
        # would error

        _plot_transformed_components_by_trialtype(transformed,
                                                  k,
                                                  data,
                                                  figdir=figdir,
                                                  adderror=True,
                                                  stderror=True,
                                                  plotting='all',
                                                  custom_name_component=custom_name_component,
                                                  )
        return

    # Plot transformed data component-wise, but for left and right epochs
    # separately.
    left, right = _get_trial_indicators(transformed, data, type='choice')
    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space for left & '
                              'right choices, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per"
                          f"-comp_leftvsright.png",
                     figdir=figdir,
                     npalette=k * 2,
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
                       label=f'component {i + 1}, {choice} choice')
            if adderror:
                ax[i].fill_between(range(len(mean)),
                                   mean - std,
                                   mean + std,
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
                           group=None,
                           label=None):
    """Aggregate data with trial structure into a temporary structure
    :param order:
    :param data:
    :param title:
    :param name:
    :param transformed:
    :param figdir: str, path to directory for figures
    :param k: int, number of components of the shared response model
    :param stderror: bool, if True, will use standard error instead of std
    :param adderror: bool, if True, will add error margins around the mean
    :param group:
    :param label:
    """
    # get ids for each subject and trialtype - the number of trialtypes differs
    # between subjects, and we need the ids to subset the consecutive list of
    # them in 'transformed'
    ids = {}
    for sub in transformed.keys():
        ids[sub] = {}
        i = 0
        # the trialtypes are consecutive in this order
        for trial in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            ids[sub][trial] = (i, i + len(data[sub][trial]))
            i += len(data[sub][trial])
    if not group:
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
                lab = f'{label[colorid] if label else trials}, k={i + 1}'
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

    if group == 'choice':
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
                        # mid = my-id (to not shadow built-in id)
                        mid = ids[sub][t]
                        assert mid[0] < mid[1]
                        for c in range(k):
                            data = \
                                [d for idx, d in enumerate(transformed[sub][c])
                                 if idx in choiceids[sub]
                                 and (mid[1] <= idx >= mid[0])]
                            tmp_transformed[sub].setdefault(c, []).extend(data)

                for comp in range(k):
                    mean, std = \
                        _get_mean_and_std_from_transformed(tmp_transformed,
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
                        ax[comp].fill_between(range(len(mean)),
                                              mean - std,
                                              mean + std,
                                              alpha=0.1,
                                              color=palettes[colorid][cid])
                cid += 1
                # Finally, add the legend.
            for a in ax:
                a.legend(loc='upper right',
                         prop={'size': 6})
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(fname)

    if group == 'prev_choice':
        # Choice of the previous trial
        prev_right = {}
        prev_left = {}
        for sub in transformed.keys():
            prev_right[sub] = []
            prev_left[sub] = []
            i = 0
            for trial in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                for epoch in data[sub][trial]:
                    if epoch['prevchoice'] == 2:
                        prev_right[sub].append(i)
                    elif epoch['prevchoice'] == 1:
                        prev_left[sub].append(i)
                    i += 1

        # special case: we group by current and previous trial choice
        if order == 'choice':
            # don't plot trialtypes or characteristics, just choice behavior
            palette, fig, ax, fname = \
                _plot_helper(k,
                             suptitle=title,
                             name=name,
                             figdir=figdir,
                             npalette=2,
                             palette='rocket'
                             )
            palette2 = sns.color_palette('mako', 2)
            palettes = [palette, palette2]
            for colorid, (prevchoiceids, choiceids) in \
                    enumerate([(prev_left, left), (prev_right, right)]):
                cid = 0
                tmp_transformed = {}
                for plotting in ['same', 'different']:
                    for sub in transformed.keys():
                        tmp_transformed[sub] = {}
                        for c in range(k):
                            if plotting == 'same':
                                choice = \
                                    [d for idx, d in
                                     enumerate(transformed[sub][c])
                                     if idx in choiceids[sub]
                                     and idx in prevchoiceids[sub]]
                            elif plotting == 'different':
                                # the logic below is slightly flawed; if a left
                                # or right choice id is not among the same
                                # previous choice id we say its a different
                                # choice - this latter category will thus
                                # include the first trial
                                choice = \
                                    [d for idx, d in
                                     enumerate(transformed[sub][c])
                                     if idx in choiceids[sub]
                                     and idx not in prevchoiceids[sub]]
                            tmp_transformed[sub].setdefault(c, []).extend(
                                choice)

                    for comp in range(k):
                        mean, std = _get_mean_and_std_from_transformed(
                            tmp_transformed,
                            comp,
                            stderror=stderror
                        )
                        ax[comp].plot(mean,
                                      color=palettes[colorid][cid],
                                      label=f'choice={colorid} ({plotting} '
                                            f'choice as previous trial), '
                                            f'k={comp + 1}')
                        if adderror:
                            ax[comp].fill_between(range(len(mean)), mean - std,
                                                  mean + std,
                                                  alpha=0.1,
                                                  color=palettes[colorid][cid])
                    cid += 1
                    # Finally, add the legend.
                for a in ax:
                    a.legend(loc='upper right',
                             prop={'size': 6})
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(fname)
        # else: trial characteristics by prev. choice
        else:
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
            for colorid, choiceids in enumerate([prev_left, prev_right]):
                cid = 0
                for trial in order:
                    print('Trial is', trial)
                    tmp_transformed = {}
                    for sub in transformed.keys():
                        tmp_transformed[sub] = {}
                        for t in trial:
                            # mid = my-id (haha), to not shadow built-in id
                            mid = ids[sub][t]
                            assert mid[0] < mid[1]
                            for c in range(k):
                                data = [d for idx, d in
                                        enumerate(transformed[sub][c])
                                        if idx in choiceids[sub] and (
                                                mid[1] <= idx >= mid[0])]
                                tmp_transformed[sub].setdefault(c, []).extend(
                                    data)

                    for comp in range(k):
                        mean, std = _get_mean_and_std_from_transformed(
                            tmp_transformed,
                            comp,
                            stderror=stderror
                        )
                        if label:
                            tid = order.index(trial)
                            lab = f'previous choice={colorid}, {label[tid]}, ' \
                                  f'k={comp + 1}'
                        else:
                            lab = f'previous choice={colorid}, trial {trial}, '\
                                  f'k={comp + 1} '
                        ax[comp].plot(mean,
                                      color=palettes[colorid][cid],
                                      label=lab)
                        if adderror:
                            ax[comp].fill_between(range(len(mean)), mean - std,
                                                  mean + std,
                                                  alpha=0.1,
                                                  color=palettes[colorid][cid])
                    cid += 1
                    # Finally, add the legend.
                for a in ax:
                    a.legend(loc='upper right',
                             prop={'size': 6})
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(fname)
    return


# noinspection PyRedundantParentheses
def _plot_transformed_components_by_trialtype(transformed,
                                              k,
                                              data,
                                              adderror=False,
                                              stderror=False,
                                              figdir='/tmp',
                                              plotting='all',
                                              custom_name_component=None
                                              ):
    """
    Summary plotting function to orchestrate plotting of various time series
    plots of transformed data. It will go through all possible trial features
    (magnitude, probability, expected value) and create plots of transformed
    trials, color-coded by feature.
    :param transformed: dict, contains transformed data
    :param k: int, number of components in shared response model
    :param data:
    :param adderror: bool, if true, add error range to mean
    :param stderror: bool, if true, use std error instead of std for error range
    :param figdir: str, path to figure directory
    :param plotting: str, what to plot. Creates every possible plot with 'all'
    :return:
    """
    # define general order types
    magnitude_order = [('A', 'B'), ('C', 'D'), ('E', 'F', 'G'), ('H', 'I')]
    magnitude_labels = [('0.5 reward'), ('1 reward'), ('2 rewards'),
                        ('4 reward')]
    trialorder = [('A'), ('B'), ('C'), ('D'), ('E'), ('F'), ('G'), ('H'), ('I')]
    probability_order = [('E', 'H'), ('C', 'F', 'I'), ('A', 'G'), ('B', 'D')]
    probability_labels = [('10% chance'), ('20% change'), ('40% chance'),
                          ('80% chance')]
    exceptedvalue_order = [('A', 'C', 'E'), ('B', 'F', 'H'), ('D', 'G', 'I')]
    expectedvalue_labels = [('0.2 EV'), ('0.4 EV'), ('0.8 EV')]
    if plotting in ('puretrialtype', 'all'):
        _plot_fake_transformed(
            order=trialorder,
            data=data,
            title="Transformed components, per trial type",
            name=f"trialtype-wise_avg-signal_shared-shape_spectral-srm_{k}"
                 f"-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
        )

    if plotting in ('magnitude', 'all'):
        # plot according to magnitude bins
        _plot_fake_transformed(
            order=magnitude_order,
            data=data,
            label=magnitude_labels,
            title="Transformed components, with trials grouped into magnitude "
                  "bins",
            name=f"trialtype-magnitude_avg-signal_shared-shape_spectral-srm_{k}"
                 f"-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
        )

    if plotting in ('probability', 'all'):
        # plot according to probability bins
        _plot_fake_transformed(
            order=probability_order,
            data=data,
            label=probability_labels,
            title="Transformed components, with trials grouped into "
                  "probability bins",
            name=f"trialtype-probability_avg-signal_shared-shape_spectral-srm_"
                 f"{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror
        )

    if plotting in ('expectedvalue', 'all'):
        # plot according to expected value
        _plot_fake_transformed(
            order=exceptedvalue_order,
            data=data,
            label=expectedvalue_labels,
            title="Transformed components, with trials grouped into expected "
                  "value bins",
            name=f"trialtype-exp-value_avg-signal_shared-shape_spectral-srm_{k}"
                 f"-feat_per-comp{custom_name_component}.png",
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
                         suptitle='Transformed components, with trials '
                                  'grouped by eventual response',
                         name=f'event-choice_avg-signal_shared-shape_spectral'
                              f'-srm_{k}-feat_per-comp{custom_name_component}.png',
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
        _plot_fake_transformed(
            order=magnitude_order,
            data=data,
            label=magnitude_labels,
            title="Transformed components, with trials grouped into magnitude "
                  "bins by eventual choice",
            name=f"trialtype-magnitude-bychoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='choice'
        )

    if plotting in ('probability-by-choice', 'all'):
        # plot according to magnitude bins
        _plot_fake_transformed(
            order=probability_order,
            data=data,
            label=probability_labels,
            title="Transformed components, with trials grouped into "
                  "probability bins by eventual choice",
            name=f"trialtype-probability-bychoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='choice'
        )

    if plotting in ('expected-value-by-choice', 'all'):
        # plot according to expected value bins
        _plot_fake_transformed(
            order=exceptedvalue_order,
            data=data,
            label=expectedvalue_labels,
            title="Transformed components, with trials grouped into expected "
                  "value bins by eventual choice",
            name=f"trialtype-expectedvalue-bychoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='choice'
        )

    if plotting in ('choice-by-previous-choice', 'all'):
        # plot according to magnitude bins
        _plot_fake_transformed(
            order='choice',
            data=data,
            title="Transformed components, with trials grouped into eventual "
                  "choice by previous choice",
            name=f"trialtype-choice-byprevchoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='prev_choice'
        )

    if plotting in ('magnitude-by-previous-choice', 'all'):
        # plot according to expected value bins
        _plot_fake_transformed(
            order=magnitude_order,
            data=data,
            label=magnitude_labels,
            title="Transformed components, with trials grouped into magnitude "
                  "bins by previous choice",
            name=f"trialtype-magnitude-byprevchoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='prev_choice'
        )

    if plotting in ('probability-by-previous-choice', 'all'):
        # plot according to expected value bins
        _plot_fake_transformed(
            order=probability_order,
            data=data,
            label=probability_labels,
            title="Transformed components, with trials grouped into "
                  "probability bins by previous choice",
            name=f"trialtype-probability-byprevchoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='prev_choice'
        )

    if plotting in ('expected-value-by-previous-choice', 'all'):
        # plot according to expected value bins
        _plot_fake_transformed(
            order=exceptedvalue_order,
            data=data,
            label=expectedvalue_labels,
            title="Transformed components, with trials grouped into excepted "
                  "value bins by previous choice",
            name=f"trialtype-expectedvalue-byprevchoice_avg-signal_shared"
                 f"-shape_spectral-srm_{k}-feat_per-comp{custom_name_component}.png",
            transformed=transformed,
            k=k,
            figdir=figdir,
            adderror=adderror,
            stderror=stderror,
            group='prev_choice'
        )


# noinspection PyUnresolvedReferences
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
    RT = [np.round(epoch['RT'] * freq)
          for subject in data for epoch in data[subject]]
    # time window centered around the reaction
    win = window * freq

    palette, fig, ax, fname = \
        _plot_helper(k,
                     suptitle='Average signal in shared space, '
                              'response-locked, component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per"
                          f"-comp_response-locked.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win / 2
                     )

    for i in range(k):
        comp = []
        for sub in transformed.keys():
            comp.extend(transformed[sub][i])

        d = [comp[idx][int(rt - win / 2):int(rt + win / 2)]
             for idx, rt in enumerate(RT) if not np.isnan(rt)]
        # if an epoch does not have enough data (too short), don't use it
        d_long_enough = np.asarray([lst for lst in d if len(lst) == win])
        mean = np.mean(d_long_enough, axis=0)
        ax[i].plot(mean, color=palette[i], label=f'component {i + 1}')
        if adderror:
            if stderror:
                std = np.std(d_long_enough, axis=0, ddof=1) / \
                      np.sqrt(d_long_enough.shape[0])
            else:
                std = np.std(d_long_enough, axis=0)
            ax[i].fill_between(range(len(std)), mean - std, mean + std,
                               alpha=0.3,
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
                     suptitle='Average signal in shared space, '
                              'response-locked, left vs. right, '
                              'component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per"
                          f"-comp_response-locked_leftvsright.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win / 2
                     )
    b = 0
    for choice, ids in [('left', left), ('right', right)]:
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
                       label=f'component {i + 1}, {choice} choice')
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
                     suptitle='Average signal in shared space, '
                              'response-locked, brainer vs nobrainer, '
                              'component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per"
                          f"-comp_response-locked_brainervsnobrainer.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win / 2
                     )
    b = 0

    for choice, ids in [('brainer', brainer), ('nobrainer', nobrainer)]:
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
                       label=f'component {i + 1}, {choice} trials')
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
                     suptitle='Average signal in shared space, '
                              'response-locked, pos vs neg feedback, '
                              'component-wise',
                     name=f"avg-signal_shared-shape_spectral-srm_{k}-feat_per"
                          f"-comp_response-locked_feedback.png",
                     figdir=figdir,
                     npalette=k * 2,
                     vline=win / 2
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
                       label=f'component {i + 1}, {choice}')
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


def plot_model_basis_topographies(datadir, model, figdir):
    """
    Take the subject-specific basis for each component of a trained SRM model
    and plot their topography.
    :return:
    """
    # use real data to create a fake evoked structure
    fname = Path(datadir) / f'sub-001/meg' / f'sub-001_task-memento_proc-sss_meg.fif'
    raw = mne.io.read_raw_fif(fname)
    # drop all non-meg sensors from the info object
    picks = raw.info['ch_names'][3:309]
    raw.pick_channels(picks)

    for subject in range(len(model.w_)):
        basis = model.w_[subject]
        k = basis.shape[1]
        fig, ax = plt.subplots(1, k)
        for c in range(k):
            # plot transformation matrix
            data = basis[:, c].reshape(-1, 1)
            fake_evoked = mne.EvokedArray(data, raw.info)
            fig = fake_evoked.plot_topomap(times=0,
                                           title=f'Subject {subject + 1}',
                                           colorbar=False,
                                           axes=ax[c], size=2
                                           )
            fname = _construct_path([
                Path(figdir),
                "group",
                "meg",
                f"viz-model-{k}_comp-{c}_sub-{subject + 1}.png"])
            fig.savefig(fname)


def plot_many_distance_matrices(results,
                                triallength,
                                figdir,
                                subjects,
                                trialorder,
                                description,
                                subset=None):
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
        title = f'Correlation distance of trialtypes ({triallength * 10}ms) \n'\
                f'in shared space ({n} components), fit on left and right \n' \
                f'averaged training data. Trials ordered by {description}'# Created: {timestr}'
        name = f'corr-dist_avg-traindata_full-stim_{n}-components_order-{description}.png'
        models[n]['full'] = plot_trialtype_distance_matrix(
            results['averaged_trials']['train']['full'],
            n,
            figdir=figdir,
            triallength=triallength,
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name,
            trialorder=trialorder,
            subset=subset
        )
        # This fits a probabilistic SRM with n features and returns the model
        # Based on the shared response space of the model, it plots the
        # correlation distance between all combinations of trial types in the
        # data (here: left visual stimulation only, averaged)
        # Importantly the plot is scaled (clim) to enhance correlation patterns
        # RELEVANCE: This plot shows whether trial information from the
        # experiment structure is preserved in the model build from averaged
        # trials
        title = f'Correlation distance of trialtypes ({triallength * 10}ms) \n'\
                f'in shared space ({n} components), fit on left  \n' \
                f'averaged training data. Trials ordered by {description}'# Created: {timestr}'
        name = f'corr-dist_avg-traindata_left-stim_{n}-components_order-{description}.png'
        models[n]['left'] = plot_trialtype_distance_matrix(
            results['averaged_trials']['train']['left'],
            n,
            figdir=figdir,
            trialtypes=9,
            clim=None,
            triallength=triallength,
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name,
            trialorder=trialorder,
            subset=subset
        )
        # This fits a probabilistic SRM with n features and returns the model
        # Based on the shared response space of the model, it plots the
        # correlation distance between all combinations of trial types in the
        # data (here: left and right visual stimulation, original time series
        # (not-averaged!))
        # RELEVANCE: This plot shows whether trial information from the
        # experiment structure is preserved in the model build from individual
        # trials
        title = f'Correlation distance of trialtypes ({triallength * 10}ms) \n'\
                f'in shared space ({n} components), fit on left and right \n' \
                f'original training data. Trials ordered by {description}'# Created: {timestr}'
        name = f'corr-dist_orig-traindata_full-stim_{n}-components_order-{description}.png'
        plot_trialtype_distance_matrix(
            results['original_trials']['train']['full'],
            n,
            figdir=figdir,
            trialtypes=234, # this ntrain * trialtypes * 2 (left and right)
            triallength=triallength,
            title=title,
            y_label='trialtype',
            x_label='trialtype',
            name=name,
            trialorder=trialorder,
            subset=subset
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
        timestr=timestr,
        trialorder=trialorder,
        description=description,
        subset=subset
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
        timestr=timestr,
        trialorder=trialorder,
        description=description,
        subset=subset
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
            trialorder=trialorder,
            description=description,
            subset=subset
        )
        # average the transformed time series across subjects, build a single
        # distance matrix from this
        # noinspection PyUnresolvedReferences
        mean_shared_test = np.mean(shared_test, axis=0)
        assert mean_shared_test.shape[0] == n
        # make the distance matrix
        # this plot displays correlation distance between data from different
        # trials in the experiment, computed from unseen test data that was
        # transformed into the shared space.
        title = f"Trialtype-by-trialtype distance between averaged \n" \
                f"transformed ({n} components) test data. Trials ordered by {description}"
        name = \
            f'group_avg-transformed_transformed-n-{n}-avg-test_trialdist-18_order-{description}.png'
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
            name=name,
            trialorder=trialorder,
            subset=subset
        )
        #logging.info(f'Permutation test on transformed test data with {n} '
        #             f'components:')
        #correlate_distance_matrix_quadrants(
        #    mean_shared_test_dist,
        #    figdir=figdir + '/group/meg',
        #    name=f'averaged_transformed_test_data_srm{n}.png')
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
            trialorder=trialorder,
            description=description,
            subset=subset
        )
        # average the transformed time series across subjects, build a single
        # distance matrix from this
        # noinspection PyUnresolvedReferences
        mean_shared_train = np.mean(shared_train, axis=0)
        assert mean_shared_train.shape[0] == n
        # make the distance matrix
        title = f"Trialtype-by-trialtype distance between averaged \n" \
                f"transformed ({n} components) train data. Trials ordered by {description}"
        name = \
            f'group_avg-transformed_transformed-n-{n}-avg-train_' \
            f'trialdist-18_order-{description}.png'
        plot_trialtype_distance_matrix(
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
            name=name,
            trialorder=trialorder,
            subset=subset
        )
        #logging.info(f'Permutation test on transformed train data with {n} '
        #             f'components:')
        #correlate_distance_matrix_quadrants(
        #    mean_shared_test_dist,
        #    figdir=figdir + '/group/meg',
        #    name=f'averaged_transformed_train_data_srm{n}.png'
        #)
    # finally, average non-transformed timeseries over subjects,
    # and build distance matrices
    for (data, label) in \
            [(results['averaged_trials']['train']['full'], 'avg-raw-train'),
             (results['averaged_trials']['test']['full'], 'avg-raw-test')]:
        # noinspection PyUnresolvedReferences
        mean = np.mean(data, axis=0)
        assert mean.shape[0] == 306
        title = f'Trialtype-by-trialtype correlation distance on \n' \
                f'{label} data. Trials ordered by {description}'#Created: {timestr}'
        name = f'groupavg_{label}_trialdist-18_order-{description}.png'
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
                                       name=name,
                                       trialorder=trialorder,
                                       subset=subset
                                       )
    return models


def correlate_distance_matrix_quadrants(distmat, figdir, name):
    """
    Take the lower triangle matrix of each quadrant in the distance matrix,
    and then correlate all quadrants lower triangle matrices with each other.
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
    a, b, c, d = \
        distmat[:9, :9], distmat[9:, :9], distmat[:9, 9:], distmat[9:, 9:]
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
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}
    plt.rc('font', **font)
    for quad_combination, idx in [('ad', 0), ('ab', 1), ('ac', 2), ('db', 3),
                                  ('dc', 4)]:
        dist = [perm[idx] for perm in permutations]
        perc = percentileofscore(dist, observed_correlations[idx])
        results[quad_combination] = perc
        # plot a histogram
        fig = sns.histplot(dist, kde=True)
        plt.axvline(observed_correlations[idx], c='black', ls='--', lw=1)
        plt.title(f'Distribution of random correlations between quadrants \n'
                  f'{quad_combination[0]} and {quad_combination[1]}, '
                  f'and observed correlation')
        plt.text(observed_correlations[idx] - 0.25, 300,
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


def compute_raw_distances(data,
                          subjects,
                          figdir,
                          trialtypes,
                          triallength,
                          feature=None,
                          nametmpl='group_task-memento_raw_avg.png',
                          y_label='trialtype',
                          x_label='trialtype',
                          subjecttitle=None,
                          grouptitle=None,
                          timestr=None,
                          trialorder=None,
                          description=None,
                          subset=None
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
    :param nametmpl:
    :param x_label:
    :param y_label:
    :param subjecttitle:
    :param grouptitle:
    :param timestr:
    :return:
    """
    # create & plot distance matrices of raw data per subject. Return dist_mat
    distmat = {}
    prev_subjecttitle = None
    for idx, sub in enumerate(subjects):
        if subjecttitle is None or subjecttitle == prev_subjecttitle:
            subjecttitle = f'Correlation distance of trialtypes ' \
                           f'({triallength * 10}ms) \n in {feature} data of' \
                           f' subject {sub}. Trials ordered by {description}'#Created: {timestr}'
            # the following is necessary to update the subject id in the title
            prev_subjecttitle = subjecttitle
        ttype = 'full-stim' if trialtypes in [18, 270] else 'left-stim'
        name = f'sub-{sub}_corr-dist_{feature}data_{ttype}_order-{description}.png'
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
                                           name=name,
                                           trialorder=trialorder,
                                           subset=subset
                                           )
    # Fisher-z transform the matrices
    zdistmat = {}
    for sub in subjects:
        # transform data from correlation distance back to correlation
        corrdist = 1 - distmat[sub]
        assert (corrdist >= -1).all() & (corrdist <= 1).all(), \
            "We have impossible correlations"
        # fisher z-transform
        print(sub)
        zdistmat[sub] = np.arctanh(corrdist)
    # average the matrices
    # noinspection PyUnresolvedReferences
    avg = np.mean(np.array([v for k, v in zdistmat.items()]), axis=0)
    # transform it back to correlation distance
    # TODO: backcheck wither this is correct!
    avg = np.nan_to_num(avg, posinf=1)
    avg_corrdist = 1 - avg
    assert not (avg_corrdist < 0).any()
    assert not (avg_corrdist >= 2).any()
    # plot it
    # set font specifications for plots
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 50}
    plt.rc('font', **font)
    plt.figure(figsize=[50, 50])
    plt.imshow(avg_corrdist, cmap='BrBG')
    plt.colorbar()
    # set a figure title according to the number of trialtypes plotted
    ttype = 'left and right' if trialtypes in [18, 270] else 'left'
    if grouptitle is None:
        grouptitle = f"Average of subject-wise {feature} data trial \n" \
                     f"distances for {ttype} stimulation. Created: {timestr}"
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
                                   name=None,
                                   trialorder=None,
                                   subset=None):
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
    :param title:
    :param x_label:
    :param y_label:
    :param name:
    :param subset: int, instead of the full data the srm was fitted on, only a
    subset (in samples) from the start of the time series is used to calculate
    correlation distance (e.g., first 100 samples)
    :return:
    """
    if clim is None:
        # default to a range of 0 to 1
        clim = [0, 1]
    if on_model_data:
        assert len(data) > 1
        assert isinstance(data, list)
        # fit a probabilistic SRM
        from pymento_meg.srm.srm import shared_response
        model = shared_response(data, features=n)
        # get the componentXtime series of each trial in SRM, and put it into
        # a nested array.
        if subset is None:
            assert triallength * trialtypes == model.s_.shape[1], \
                "mismatch between the specified length of a stimulus " \
                "presentation (triallength) and the available data. Please check!"
            trialmodels_ = np.array(
                [model.s_[:, triallength * i:triallength * (i + 1)].ravel()
                 for i in range(trialtypes)])
        else:
            trialmodels_ = np.array(
                [model.s_[:, triallength * i:triallength * i + subset].ravel()
                 for i in range(trialtypes)])
    else:
        assert isinstance(data, np.ndarray)
        if subset is None:
            assert triallength * trialtypes == data.shape[1], \
                "mismatch between the specified length of a stimulus " \
                "presentation (triallength) and the available data. Please check!"
            trialmodels_ = np.array(
                [data[:, triallength * i:triallength * (i + 1)].ravel()
                 for i in range(trialtypes)])
        else:
            trialmodels_ = np.array(
                [data[:, triallength * i:triallength * i + subset].ravel()
                 for i in range(trialtypes)])

    dist_mat = sp_distance.squareform(
        sp_distance.pdist(trialmodels_, metric='correlation'))
    # correlation distance is between 0 (perfect correlation) and 2
    # (perfect anticorrelation)
    assert np.logical_and(dist_mat >= 0, dist_mat <= 2).all()
    # set font specifications for plots
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 50}
    plt.rc('font', **font)
    plt.figure(figsize=[50, 50])
    ax = plt.gca()
    im = plt.imshow(dist_mat, cmap='BrBG', clim=[0, 2])

    if title is None:
        # set a figure title according to the number of trialtypes plotted
        ttype = 'left and right' if trialtypes in [18, 270] else 'left'
        if on_model_data:
            title = f"Correlation distance between trialtypes for \n" \
                    f"{ttype} stimulation"
        else:
            title = f"Correlation distance between trialtypes for \n" \
                    f"{ttype} stimulation in raw data"
    title = title
    plt.title(title)
    if x_label is None:
        x_label = 'trial type'
    if y_label is None:
        y_label = 'trial type'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if trialorder is not None:
        fctr = int(trialtypes / len(trialorder))
        plt.xticks(ticks=range(len(trialorder)*fctr), labels=trialorder*fctr)
        plt.yticks(ticks=range(len(trialorder)*fctr), labels=trialorder*fctr)
    if name is None:
        datatype = 'srm' if on_model_data else 'raw'
        name = f'trialtype-distance_{datatype}-data_{trialtypes}-trials_n-{n}' \
               f'_feature-{feature}.png '
    fname = _construct_path([Path(figdir),  f'group/meg' , name])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    logging.info(f"Saving a distance matrix to {fname}")
    plt.savefig(fname)
    plt.close('all')
    if on_model_data:
        return model
    else:
        return dist_mat


def plot_distance_matrix(model, idx, figdir, freq, subject, condition, timespan):
    """
    plot a distance matrix between time points from the shared response.
    :param model:
    :param idx
    :param figdir:
    :return:
    """
    dist_mat = sp_distance.squareform(sp_distance.pdist(model.s_.T,
                                                        metric='correlation'))
    plt.xlabel(f't ({freq} = 1sec)')
    plt.ylabel(f't ({freq} = 1sec)')
    plt.imshow(dist_mat, cmap='BrBG',  clim=[0, 2])
    # TODO: maybe add vertical lines in experiment landmarks
    plt.colorbar()
    fname = _construct_path([Path(figdir), f'sub-{subject}', 'meg',
                             f'sub-{subject}_task-memento_srm-{idx}_{condition}-{timespan}_distances.png'])
    plt.savefig(fname)
    plt.close()


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
    logging.info('Plotting the data transformed with the SRM model.')
    title = 'Full trial duration' if timespan == 'fulltrial' else \
        'Duration of the first stimulus' if timespan == 'firststim' else \
            '400ms +/- decision time' if timespan == 'decision' else None
    # TODO: this needs some indication of which subjects the plot is made from
    sns.set(rc={'figure.figsize': (8, 4)})
    for i in range(nfeatures):
        if type(subject) == list:
            if len(subject) == 1:
                subject = subject[0]
            else:
                subject = 'group'
        fname = _construct_path([Path(figdir), f'sub-{subject}', 'meg',
                                 f'sub-{subject}_{mdl}_{nfeatures}-feat_task-{cond}_model-{timespan}_comp_{i}.png'])
        ax = sns.lineplot(data=df, y=i, x=df.index, hue='trial_type')
        ax.set(xlabel='samples', ylabel=f'comp. {i+1} (a.U.)', ylim=(-10, 10), title=title)
        if timespan == 'fulltrial':
            # define the timing of significant events in the trial time course:
            # onset and offset of visual stimuli
            events = [0, 0.7 * freq, 2.7 * freq, 3.4 * freq]
            # plot horizontal lines to mark the end of visual stimulation
            [pylab.axvline(ev, linewidth=1, color='grey', linestyle='dashed')
             for ev in events]
        if timespan == 'decision':
            # mark the decision
            pylab.axvline(0.4 * freq, color='grey', linestyle='dashed')
            locations, labels = plt.xticks()
            plt.xticks(locations, [str(int(l)) for l in locations - 400])
        plt.tight_layout()
        plot = ax.get_figure()
        logging.info(f'Saving figure {fname}')
        plot.savefig(fname)
        plot.clear()
        plt.rcParams.update(plt.rcParamsDefault)
