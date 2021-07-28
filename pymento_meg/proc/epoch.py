import logging
import pandas as pd
from pathlib import Path
from pymento_meg.utils import (
    _plot_evoked_fields,
    _get_channel_subsets,
)


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def get_stimulus_characteristics(subject,
                                 bids_path,
                                 columns=['trial_no', 'LoptMag', 'LoptProb'],
                                 epochs=None,
                                 **kwargs):
    """
    Retrieve metadata about the stimulus/trial properties recorded in the
    logdata.
    Add metadata on reward magnitude and probability to epochs of first visual
    events.
    Event name ->   description ->      count:
    lOpt1 -> LoptMag 0.5, LoptProb 0.4 -> 70
    lOpt2 -> LoptMag 0.5, LoptProb 0.8 -> 65
    lOpt3 -> LoptMag 1, LoptProb 0.2 -> 50
    lOpt4 -> LoptMag 1, LoptProb 0.8 -> 70
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

    :param bids_path: str, path to the bids directory with logfiles
    :param subject: str, subject identifier
    :param epochs: epochs, optional; this is where trial information is added
    to as metadata
    :param columns: list of column names to index
    """
    assert type(columns) == list
    # get a dataframe of the visual features
    metadata = get_trial_features(bids_path, subject, columns)
    if epochs:
        # if we got an epochs object, add the trial properties as metadata
        # TODO: this probably needs a length check...
        # This can then be indexed/queried like this:
        # visuals['LoptMag == 1.0'] or visuals['LoptMag == 4 and LoptProb == 0.8']
        epochs.metadata = metadata
        return epochs
    # if we don't attach this information to epochs, return it as a dataframe.
    return metadata


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
    logging.info(f'Retrieving Trial metadata for subject sub-{subject}, from the '
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

