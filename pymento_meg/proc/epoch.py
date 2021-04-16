import mne
import pandas as pd
from pathlib import Path
from pymento_meg.config import reject_criteria
from pymento_meg.proc.bids import get_events
from pymento_meg.utils import (
    _plot_evoked_fields,
    _get_channel_subsets,
)


def add_metadata_to_visuals(visuals, subject, bids_path, **kwargs):
    """
    Add metadata on reward magnitude and probability to epochs of first visual
    events.
    Event name ->   description ->      count:
    lOpt1 -> LoptMag 0.5, LoptProb 0.4 -> 70
    lOpt1 -> LoptMag 0.5, LoptProb 0.8 -> 65
    lOpt3 -> LoptMag 1, LoptProb 0.2 -> 50
    lOpt4 -> LoptMag 1, LoptProb 0.8 -> 70
    lOpt5 -> LoptMag 2, LoptProb 0.1 -> 50
    lOpt5 -> LoptMag 2, LoptProb 0.1 -> 50
    lOpt6 -> LoptMag 2, LoptProb 0.2 -> 35
    lOpt7 -> LoptMag 2, LoptProb 0.4 -> 50
    lOpt8 -> LoptMag 4, LoptProb 0.1 -> 70
    lOpt9 -> LoptMag 4, LoptProb 0.2 -> 50

    No epochs for:
    LoptMag 0.5, LoptProb 0.1
    LoptMag 0.5, LoptProb 0.2
    LoptMag 1, LoptProb 0.1
    LoptMag 1, LoptProb 0.4
    LoptMag 2, LoptProb 0.8
    LoptMag 4, LoptProb 0.4
    LoptMag 4, LoptProb 0.8

    :param visuals: epochs, visualevents of left stimulation
    """

    # get a dataframe of the visual features
    metadata = get_trial_features(bids_path, subject, ['LoptMag', 'LoptProb'])
    visuals.metadata = metadata
    # This can now be indexed/queried like this:
    # visuals['LoptMag == 1.0'] or visuals['LoptMag == 4 and LoptProb == 0.8']
    return


def get_trial_features(bids_path, subject, column):
    """
    Get the spatial frequency and angle of the gabor patches from the log files.
    :param bids_path: str, path to BIDS directory from which we can get log files
    :param subject: str, subject identifier, takes the form '001'
    :param column: str or list, key(s) to use for indexing the logs,
    will be returned as metadata
    """
    fname = Path(bids_path) / f'sub-{subject}' / 'meg' /\
            f'sub-{subject}_task-memento_log.tsv'
    df = pd.read_csv(fname, sep='\t')
    print(f'Retrieving Trial metadata for subject sub-{subject}, from the '
          f'column(s) {column} in the file {fname}.')
    if type(column) == list:
        for c in column:
            assert c in df.keys()
    elif type(column) == str:
        assert column in df.keys()
    metadata = df[column]
    # if this is only one key, we're not a dataframe, but a series
    if isinstance(metadata, pd.Series):
        metadata = metadata.to_frame()
    assert isinstance(metadata, pd.DataFrame)
    return metadata


def epoch_data(
    raw,
    subject,
    conditionname=None,
    sensor_picks=None,
    picks=None,
    pick_description=None,
    figdir="/tmp",
    reject_criteria=reject_criteria,
    tmin=-0.2,
    tmax=0.7,
    reject_bad_epochs=True,
    autoreject=False,
):
    """
    Create epochs from specified events.
    :param tmin: int, start time before event. Defaults to -0.2 in MNE 0.23dev
    :param tmax: int, end time after event. Defaults to 0.5 in MNE 0.23dev
    :param sensor_picks: list, sensors that should be plotted separately
    :param picks: list, sensors that epoching should be restricted to
    :param pick_description: str, a short description (no spaces) of the picks,
    e.g., 'occipital' or 'motor'.
    :param figdir: str, Path to where diagnostic plots should be saved.

    TODO: we could include a baseline correction
    TODO: figure out projections -> don't use if you can use SSS
    TODO: autoreject requires picking only MEG channels in epoching
    """

    events, event_dict = get_events(raw)

    epoch_params = {
        "raw": raw,
        "events": events,
        "event_id": event_dict,
        "tmin": tmin,
        "tmax": tmax,
        "preload": True,
        "on_missing": "warn",
        "verbose": True,
        "picks": 'meg',     # needed
    }

    if reject_bad_epochs and not autoreject:
        # we can reject based on predefined criteria. Add it as an argument to
        # the parameter list
        epoch_params["reject"] = reject_criteria
    else:
        epoch_params["reject"] = None

    epochs = mne.Epochs(**epoch_params)
    if reject_bad_epochs and not autoreject:
        epochs.plot_drop_log()
    if autoreject:
        # if we want to perform autorejection of epochs using the
        # autoreject tool
        for condition in conditionname:
            # do this for all relevant conditions
            epochs = autoreject_bad_epochs(epochs=epochs,
                                           key=condition)

    #for condition in conditionname:
    #    _plot_epochs(
    #        raw,
    #        epochs=epochs,
    #        subject=subject,
    #        key=condition,
    #        figdir=figdir,
    #        picks=sensor_picks,
    #        pick_description=pick_description,
    #    )
    return epochs


def _plot_epochs(
    raw,
    epochs,
    subject,
    key,
    figdir,
    picks,
    pick_description,
):
    """

    TODO: decide for the right kinds of plots, and whether to plot left and right
    seperately,
    :param picks: list, all channels that should be plotted. You can also select
    predefined locations: lpar, rpar, locc, rocc, lfro, rfro, ltem, rtem, ver.
    :param pick_description: str, a short description (no spaces) of the picks,
    e.g., 'occipital' or 'motor'.
    """
    # subselect the required condition.
    # For example visuals = epochs['visualfirst']
    wanted_epochs = epochs[key]
    average = wanted_epochs.average()
    # Some general plots over all channels
    _plot_evoked_fields(
        data=average, subject=subject, figdir=figdir, key=key, location="avg-epoch-all"
    )
    if picks:
        # If we want to plot a predefined sensor space, e.g.,
        # right parietal or left temporal, load in those lists of sensors
        assert type(picks) == list
        if len(picks) >= 2:
            # more than one selection in this list
            sensors = _get_channel_subsets(raw)
            mypicklist = []
            for p in picks:
                if p in sensors.keys():
                    mypicklist.extend(sensors[p])
                else:
                    mypicklist.extend(p)
            subset = epochs.pick_channels(mypicklist)
            subset_average = subset.average()
            _plot_evoked_fields(
                data=subset_average,
                subject=subject,
                figdir=figdir,
                key=key,
                location=pick_description,
            )

            # TODO plot with pick_description

    return


def autoreject_bad_epochs(epochs, key):
    import autoreject
    import numpy as np

    # these values come straight from the tutorial:
    # http://autoreject.github.io/auto_examples/plot_auto_repair.html
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    # important: Requires epochs with only MEG sensors, selected during epoching!
    ar = autoreject.AutoReject(
        n_interpolates,
        consensus_percs,
        thresh_method="random_search",
        random_state=42,
    )
    subset = epochs[key]
    ar.fit(subset)
    epochs_clean = ar.transform(subset)
    return epochs_clean