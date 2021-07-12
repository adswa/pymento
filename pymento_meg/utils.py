#!/usr/bin/env python

import mne
import os
import logging
import numpy as np
from pathlib import Path


# Set a few global variables
# The data is not preconditioned unless this variable is reset
preconditioned = False

from .config import (
    reject_criteria,
)

# Define data processing functions and helper functions


def _get_channel_subsets(raw):
    """
    Return defined sensor locations as a list of relevant sensors
    """
    # TODO: fill in with sensor names for left and right
    parietal_left = mne.read_selection(name=["Left-parietal"], info=raw.info)
    parietal_right = mne.read_selection(name=["Right-parietal"], info=raw.info)
    occipital_left = mne.read_selection(name=["Left-occipital"], info=raw.info)
    occipital_right = mne.read_selection(name=["Right-occipital"], info=raw.info)
    frontal_left = mne.read_selection(name=["Left-frontal"], info=raw.info)
    frontal_right = mne.read_selection(name=["Right-frontal"], info=raw.info)
    vertex_sensors = mne.read_selection(name=["Vertex"], info=raw.info)
    temporal_left = mne.read_selection(name=["Left-temporal"], info=raw.info)
    temporal_right = mne.read_selection(name=["Right-temporal"], info=raw.info)
    sensors = {
        "lpar": parietal_left,
        "rpar": parietal_right,
        "locc": occipital_left,
        "rocc": occipital_right,
        "lfro": frontal_left,
        "rfro": frontal_right,
        "ltem": temporal_left,
        "rtem": temporal_right,
        "ver": vertex_sensors,
    }
    return sensors


def _check_if_bids_directory_exists(outpath):
    """
    Helper function that checks if a directory exists, and if not, creates it.
    """
    check_dir = os.path.dirname(outpath)
    logging.info(check_dir)
    if not os.path.isdir(Path(check_dir)):
        logging.info(
            f"The BIDS directory {check_dir} does not seem to exist. "
            f"Attempting creation..."
        )
        os.makedirs(Path(check_dir))


def _construct_path(components):
    """
    Helper function to construct a path to save a file or figure in, check if
    the directory exists, and create the path recursively, if necessary.

    :params components: list, path components
    """
    fpath = os.path.join(*components)
    _check_if_bids_directory_exists(fpath)
    return fpath


def repair_triggers(events, log_df):
    """
    The experiment had a mix of writing triggers in integers, and photodiode
    signals that partially overlaid these integer triggers, resulting in
    obscure trigger values such as 32780. Which trigger values were overlaid
    changed over the course of the experiment. For all subjects, trigger event
    22 (delay) and 24 (right option) were obfuscated to 32790 and 32792.
    For subjects 1 and 2, this was true also for all other Stimulus events.

    Overall, the experiment is quite messy. We should read in the experiment
    logs to make sure that MEG triggers and logs are consistent
    :param events: mne events representation
    :param log_df: pandas dataframe with log information about the experiment
    """

    # subtract the photodiode value (seems to be 0b1000000000000000, i.e.,
    # 32768 as an integer) from all obscurely large values
    renamed_triggers = [i if i < 30000 else i - 32768 for i in events[:, 2]]
    # replace the existing trigger names
    events[:, 2] = renamed_triggers

    # remove event duplications if there are two consecutive identical events
    # those occur if the photodiode duplicated a trigger signal
    # I am keeping the first occurrence of them and disregard the later
    cropped_events = []
    for idx, sample in enumerate(events):
        # append the first sample in any case
        if idx == 0:
            cropped_events.append(list(sample))
            continue
        # when the trigger name of the previous sample is identical to the
        # current sample's trigger name, don't append this sample
        if sample[2] == events[idx - 1, 2]:
            continue
        # if two consecutive samples have different triggers, append the sample
        elif sample[2] != events[idx - 1, 2]:
            cropped_events.append(list(sample))
    # now we should have the same format as events were
    cropped_events = np.asarray(cropped_events)
    if log_df is None:
        # we didn't get log files
        logging.info("Did not receive dataframe with experiment log. No sanity checks.")
        return cropped_events
    logging.info("Performing basic sanity checks based on the log files of " "the experiment")
    ev, event_counts = np.unique(cropped_events[:, 2], return_counts=True)
    events_in_data = dict(zip(ev, event_counts))
    keys_in_log = log_df.keys()
    # as many fixation crosses and feedbacks (start and end) as trials
    # subject 005 only has 509 fixation crosses!
    try:
        assert log_df["trial_no"].shape[0] == events_in_data[10] == events_in_data[27]
    except AssertionError:
        logging.info(
            f"The number of trials, fixations, and feedbacks does not match: "
            f"{log_df['trial_no'].shape[0]} vs. {events_in_data[10]} "
            f"vs {events_in_data[27]}. Subject 005 is known to have such an "
            f"issue - is it subject 005?"
        )
    # the is a variable number of "empty_screen" events in the data
    if "Empty_screen" in keys_in_log:
        # this subject has "empty screen" onsets
        assert log_df["Empty_screen"].count() == events_in_data[26]

    return cropped_events


def eventreader(raw, subject, event_dict, df, outputdir="/tmp/"):
    """
    Find all events and repair them, if necessary.
    :param raw:
    :param subject: str, subject identifier in the form of '001'
    :param event_dict: dict, trigger name associations
    :param outputdir: str, path to where diagnostic figures are saved
    :param df: dataframe, contains logging information about the experiment
    :return:
    """
    events = mne.find_events(
        raw,
        min_duration=0.002,  # ignores spurious events
        uint_cast=True,  # workaround Elekta acquisition bug, causes neg. values
        stim_channel=["STI101", "STI102", "STI016"],  # get all triggers
        consecutive=True,  # Trigger are overlayed by photodiode signals, sadly
    )
    # remove events that are known to be spurious. It is not a problem if they
    # are only present in some subjects, we can specify a group-level list here
    # 5 was photodiode onset, but it is not relevant for us
    exclusion_list = [254, 32768, 5]
    events = mne.pick_events(events, exclude=exclusion_list)
    # remove any signals obfuscated by a photodiode
    events = repair_triggers(events=events, log_df=df)

    # plot events. This works without raw data
    fig = mne.viz.plot_events(
        events,
        sfreq=raw.info["sfreq"],
        first_samp=raw.first_samp,
        event_id=event_dict,
        on_missing="warn",
    )
    fig.suptitle(
        "Full event protocol for {} ({})".format(
            raw.info["subject_info"]["first_name"],
            raw.info["subject_info"]["last_name"],
        )
    )
    fpath = _construct_path(
        [
            Path(outputdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_eventplot.png",
        ]
    )
    fig.savefig(str(fpath))
    return events

def _plot_evoked_fields(data, subject, figdir, key="unnamed", location="avg-epoch-all"):
    """
    Helper to plot evoked field with all available plots
    :return:
    """
    # make joint plot of topographies and sensors
    figpath_grad = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_{location}_cond-{key}_joint-grad.png",
        ]
    )
    figpath_mag = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_{location}_cond-{key}_joint-mag.png",
        ]
    )
    fig = data.plot_joint()
    fig1 = fig[0]
    fig2 = fig[1]
    fig1.savefig(figpath_grad)
    fig2.savefig(figpath_mag)
    # also plot topographies
    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_{location}_cond-{key}_topography.png",
        ]
    )
    fig = data.plot_topo()
    fig.savefig(figpath)
    # also save the data as an image
    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_{location}_cond-{key}_image.png",
        ]
    )
    fig = data.plot_image()
    fig.savefig(figpath)


def evoked_visual_potentials(raw, subject, event_dict, figdir="/tmp/"):
    """
    Function to plot visual potentials from the first visual option (left) as a
    sanity check
    """

    events = eventreader(
        raw=raw, subject=subject, outputdir=figdir, event_dict=event_dict
    )

    visual_epochs = epoch_data(
        raw=raw,
        events=events,
        event_dict=event_dict,
        subject=subject,
        conditionname=["visualfirst"],
        sensor_picks=["locc", "rocc"],
        picks=None,
        pick_description="occipital",
        figdir=figdir,
        tmax=0.7,
        tmin=-0.2,
        reject_criteria=reject_criteria,
        reject_bad_epochs=True,
        autoreject=False,
    )



def artifacts(raw):
    # see https://mne.tools/stable/auto_tutorials/preprocessing/plot_10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-plot-10-preprocessing-overview-py
    # low frequency drifts in magnetometers
    mag_channels = mne.pick_types(raw.info, meg="mag")
    raw.plot(
        block=True,
        duration=60,
        order=mag_channels,
        n_channels=len(mag_channels),
        remove_dc=False,
    )

    # power line noise
    fig = raw.plot_psd(block=True, tmax=np.inf, fmax=250, average=True)

    # heartbeat artifacts
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))
