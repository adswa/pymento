#!/usr/bin/env python

import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib import interactive
from pathlib import Path

# plotting settings
interactive(True)
import matplotlib

matplotlib.use("Qt5Agg")
mne.set_log_level("info")

# Set a few global variables
# The data is not preconditioned unless this variable is reset
preconditioned = False

from .config import (
    channel_types,
    reject_criteria,
    flat_criteria,
    crosstalk_file,
    fine_cal_file,
    subject_list,
    event_dict,
)

# Define data processing functions and helper functions


def _filter_data(
    raw,
    l_freq=None,
    h_freq=40,
    picks=None,
    fir_window="hamming",
    filter_length="auto",
    iir_params=None,
    method="fir",
    phase="zero",
    l_trans_bandwidth="auto",
    h_trans_bandwidth="auto",
    pad="reflect_limited",
    skip_by_annotation=("edge", "bad_acq_skip"),
    fir_design="firwin",
):
    """
    Filter raw data. This is an exact invocation of the filter function of mne 0.23 dev.
    It uses all defaults of this version to ensure future updates to the defaults will not
    break the analysis result reproducibility.
    :param raw:
    :param l_freq:
    :param h_freq:
    :param fir_window:
    :param filter_length:
    :param phase:
    :param l_trans_bandwidth:
    :param h_trans_bandwidth:
    :param fir_design:
    :return:
    """
    # make sure that the data is loaded
    raw.load_data()
    raw.filter(
        h_freq=h_freq,
        l_freq=l_freq,
        picks=picks,
        filter_length=filter_length,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        iir_params=iir_params,
        method=method,
        phase=phase,
        skip_by_annotation=skip_by_annotation,
        pad=pad,
        fir_window=fir_window,
        fir_design=fir_design,
        verbose=True,
    )
    return raw


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


def _downsample(raw, frequency):
    """
    Downsample data using MNE's built-in resample function
    """
    raw_downsampled = raw.copy().resample(sfreq=frequency,
                                          verbose=True)
    return raw_downsampled


def _get_first_file(files):
    """
    Helper function to return the first split of a range of files.
    This is necessary because the file names are inconsistent across subjects.
    This function should return file names of any preprocessing flavor or Raw
    directory in the correct order for reading in.

    :param files: list of str, with file names
    :return:

    """
    first, second, third = None, None, None
    import os.path as op

    # basic sanity check:
    # there should be three fif files
    assert len(files) == 3
    # check if there are any files starting with a number
    starts_with_digit = [op.basename(f)[0].isdigit() for f in files]
    if not any(starts_with_digit):
        # phew, we're not super bad
        for f in files:
            # only check the filenames
            base = op.basename(f)
            if len(base.split("-")) == 2 and base.split("-")[-1].startswith("1"):
                second = f
            elif len(base.split("-")) == 2 and base.split("-")[-1].startswith("2"):
                third = f
            elif len(base.split("-")) == 1:
                first = f
            else:
                # we shouldn't get here
                raise ValueError(f"Cannot handle file list {files}")
    else:
        # at least some file names start with a digit
        if all(starts_with_digit):
            # this is simple, files start with 1_, 2_, 3_
            first, second, third = sorted(files)
        else:
            # only some file names start with a digit. This is really funky.
            for f in files:
                base = op.basename(f)
                if base[0].isdigit() and base[0] == "1" and len(base.split("-")) == 1:
                    first = f
                elif base[0].isdigit() and base[0] == "2" and len(base.split("-")) == 1:
                    second = f
                elif base[0].isdigit() and base[0] == "2" and len(base.split("-")) == 2:
                    if base.split("-")[-1].startswith("1"):
                        second = f
                    elif base.split("-")[-1].startswith("2"):
                        third = f
                elif len(base.split("-")) == 2 and base[0].isalpha():
                    if base.split("-")[-1].startswith("1"):
                        second = f
                    elif base.split("-")[-1].startswith("2"):
                        third = f
                else:
                    # this shouldn't happen
                    raise ValueError(f"Cannot handle file list {files}")
    # check that all files are defined
    assert all([v is not None for v in [first, second, third]])
    print(f"Order the files as follows: {first}, {second}, {third}")
    return first, second, third


def read_data_original(
    directory, subject, savetonewdir=False, bidsdir=None, preprocessing="Raw"
):
    """
    The preprocessed MEG data is split into three files that MNE Python
    can't automatically co-load.
    We read in all files, and concatenate them by hand.
    :param directory: path to a subject directory.
    :param subject: str, subject identifier ('001'), used for file names
     and logging
    :param savetonewdir: Boolean, if True, save the data as BIDS conform
    files into a new directory
    :param newdir: str, Path to where BIDS conform data shall be saved
    :param preprocessing: Data flavour to load. Existing directories are
     'Move_correc_SSS_realigneddefault_nonfittoiso' and 'Raw' (default)
    :return:
    """

    # We're starting with the original data from Luca. The files were
    # transferred from Hilbert as is, and have a non-BIDS and partially
    # inconsistent naming and directory structure
    # First, construct a Path to a preprocessed or Raw directory
    directory = Path(directory) / f"memento_{subject}"
    path = (
        Path(directory) / preprocessing / "*.fif"
        if preprocessing
        else Path(directory) / "*.fif"
    )
    if not os.path.exists(os.path.split(path)[0]):  # TODO: test this
        # some subjects have an extra level of directories
        path = (
            Path(directory) / "*" / preprocessing / "*.fif"
            if preprocessing
            else Path(directory) / "*" / "*.fif"
        )
    print(f"Reading files for subject sub-{subject} from {path}.")
    # file naming is a mess. We need to make sure to sort the three files
    # correctly
    unsorted_files = glob(str(path))
    first, second, third = _get_first_file(unsorted_files)
    if subject != '005':
        # subject five doesn't have consistent subject identifiers in the name.
        # automatic reading in would only load the first.
        try:
            raw = mne.io.read_raw_fif(first)
        except ValueError:
            print(
                f"WARNING Irregular file naming. Will read files in sequentially "
                f"in the following order: {first}{second}{third}"
            )
            # read the splits
            split1 = mne.io.read_raw_fif(first, on_split_missing="warn")
            split2 = mne.io.read_raw_fif(second, on_split_missing="warn")
            split3 = mne.io.read_raw_fif(third, on_split_missing="warn")
            # concatenate all three split files
            raw = mne.concatenate_raws([split1, split2, split3])
    else:
        # read the splits
        split1 = mne.io.read_raw_fif(first, on_split_missing="warn")
        split2 = mne.io.read_raw_fif(second, on_split_missing="warn")
        split3 = mne.io.read_raw_fif(third, on_split_missing="warn")
        # concatenate all three split files
        raw = mne.concatenate_raws([split1, split2, split3])
    # explicitly set channel types to EOG and ECG sensors
    raw.set_channel_types(channel_types)

    if savetonewdir:
        if not bidsdir:
            print(
                "I was instructed to save BIDS conform raw data into a"
                "different directory, but did not get a path."
            )
            return raw

        outpath = _construct_path(
            [
                Path(bidsdir),
                f"sub-{subject}",
                "meg",
                f"sub-{subject}_task-memento_meg.fif",
            ],
            subject,
        )
        print(
            f"Saving BIDS-compliant raw data from subject {subject} into " f"{outpath}"
        )
        raw.save(outpath, split_naming="bids", overwrite=True)
    return raw


def motion_estimation(subject, raw, head_pos_outdir="/tmp/", figdir="/tmp/"):
    """
    Calculate head positions from HPI coils as a prerequisite for movement
    correction.
    :param subject: str, subject identifier; used for writing file names &
    logging
    :param raw: Raw data object
    :param head_pos_outdir: directory to save the head position file to. Should
    be the root of a bids directory
    :return: head_pos: head positions estimates from HPI coils
    """
    # Calculate head motion parameters to remove them during maxwell filtering
    # First, extract HPI coil amplitudes to
    print(f"Extracting HPI coil amplitudes for subject sub-{subject}")
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    # compute time-varying HPI coil locations from amplitudes
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    print(f"Computing head positions for subject sub-{subject}")
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
    # save head positions
    outpath = _construct_path(
        [
            Path(head_pos_outdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_headshape.pos",
        ],
        subject,
    )
    print(f"Saving head positions as {outpath}")
    mne.chpi.write_head_pos(outpath, head_pos)

    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_headmovement.png",
        ],
        subject,
    )
    fig = mne.viz.plot_head_positions(head_pos, mode="traces")
    fig.savefig(figpath)
    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_headmovement_scaled.png",
        ],
        subject,
    )
    fig = mne.viz.plot_head_positions(
        head_pos, mode="traces", destination=raw.info["dev_head_t"], info=raw.info
    )
    fig.savefig(figpath)
    return head_pos


def _check_if_bids_directory_exists(outpath, subject):
    """
    Helper function that checks if a directory exists, and if not, creates it.
    """
    check_dir = os.path.dirname(outpath)
    print(check_dir)
    if not os.path.isdir(Path(check_dir)):
        print(
            f"The BIDS directory {check_dir} does not seem to exist. "
            f"Attempting creation..."
        )
        os.makedirs(Path(check_dir))


def _construct_path(components, subject):
    """
    Helper function to construct a path to save a file or figure in, check if
    the directory exists, and create the path recursively, if necessary.

    :params components: list, path components
    """
    fpath = os.path.join(*components)
    _check_if_bids_directory_exists(fpath, subject)
    return fpath


# TODO: We could do maxwell filtering without applying a filter when we remove
# chpi and line noise beforehand.
# mne.chpi.filter_chpi is able to do this
def filter_chpi_and_line(raw):
    """
    Remove Chpi and line noise from the data. This can be useful in order to
    use no filtering during bad channel detection for maxwell filtering.
    :param raw: Raw data, preloaded
    :return:
    """
    from mne.chpi import filter_chpi

    # make sure the data is loaded first
    print("Loading data for CHPI and line noise filtering")
    raw.load_data()
    print("Applying CHPI and line noise filter")
    # all parameters are set to the defaults of 0.23dev
    filter_chpi(
        raw,
        include_line=True,
        t_step=0.01,
        t_window="auto",
        ext_order=1,
        allow_line_only=False,
        verbose=None,
    )
    # the data is now preconditioned, hence we change the state of the global
    # variable
    global preconditioned
    preconditioned = True
    return raw


def maxwellfilter(
    raw,
    crosstalk_file,
    fine_cal_file,
    subject,
    headpos_file=None,
    compute_motion_params=True,
    head_pos_outdir="/tmp/",
    figdir="/tmp/",
    outdir="/tmp/",
    filtering=False,
    filter_args=None,
):
    """

    :param raw:
    :param crosstalk_file: crosstalk compensation file from the Elekta system to
     reduce interference between gradiometers and magnetometers
    :param calibration_file: site-specific sensor orientation and calibration
    :param figdir: str, path to directory to save figures in
    :param filtering: if True, a filter function is ran on the data after SSS.
    By default, it is a 40Hz low-pass filter.
    :param filter_args: dict; if filtering is True, initializes a filter with the
    arguments provided
    :return:
    """
    from mne.preprocessing import find_bad_channels_maxwell

    if not compute_motion_params:
        if not headpos_file or not os.path.exists(headpos_file):
            print(
                f"Could not find or read head position files under the supplied"
                f"path: {headpos_file}. Recalculating from scratch."
            )
            head_pos = motion_estimation(subject, raw, head_pos_outdir, figdir)
        print(
            f"Reading in head positions for subject sub-{subject} "
            f"from {headpos_file}."
        )
        head_pos = mne.chpi.read_head_pos(headpos_file)

    else:
        print(f"Starting motion estimation for subject sub-{subject}.")
        head_pos = motion_estimation(subject, raw, head_pos_outdir, figdir)

    raw.info["bads"] = []
    raw_check = raw.copy()

    if preconditioned:
        # preconditioned is a global variable that is set to True if some form
        # of filtering (CHPI and line noise removal or general filtering) has
        # been applied.
        # the data has been filtered, and we can pass h_freq=None
        print("Performing bad channel detection without filtering")
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check,
            cross_talk=crosstalk_file,
            calibration=fine_cal_file,
            return_scores=True,
            verbose=True,
            h_freq=None,
        )
    else:
        # the data still contains line noise (50Hz) and CHPI coils. It will
        # filter the data before extracting bad channels
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check,
            cross_talk=crosstalk_file,
            calibration=fine_cal_file,
            return_scores=True,
            verbose=True,
        )
    print(
        f"Found the following noisy channels: {auto_noisy_chs} \n "
        f"and the following flat channels: {auto_flat_chs} \n"
        f"for subject sub-{subject}"
    )
    bads = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
    raw.info["bads"] = bads
    # free up space
    del raw_check
    # plot as a sanity check
    for ch_type in ["grad", "mag"]:
        plot_noisy_channel_detection(
            auto_scores, ch_type=ch_type, subject=subject, outpath=figdir
        )
    print(
        f"Signal Space Separation with movement compensation "
        f"starting for subject sub-{subject}"
    )
    ## TODO: movement compensation can be done during maxwell filtering but also during
    raw_sss = mne.preprocessing.maxwell_filter(
        raw,
        cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        head_pos=head_pos,
        verbose=True,
    )
    # save sss files
    fname = _construct_path(
        [
            Path(outdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_proc-sss.fif",
        ],
        subject,
    )
    raw_sss.save(fname, split_naming="bids", overwrite=True)
    if filtering:
        print(
            f"Filtering raw SSS data for subject {subject}. The following"
            f"additional parameters were passed: {filter_args}"
        )
        raw_sss_filtered = raw_sss.copy()
        raw_sss_filtered = _filter_data(raw_sss, **filter_args)
        # TODO: Downsample
        _plot_psd(raw_sss_filtered, subject, figdir, filtering)
        return raw_sss_filtered

    _plot_psd(raw_sss, subject, figdir, filtering)
    return raw_sss


def _plot_psd(raw, subject, figdir, filtering):
    """
    Helper to plot spectral densities
    """
    print(
        f"Plotting spectral density plots for subject sub-{subject}"
        f"after Maxwell filtering."
    )
    if filtering:
        # append a 'filtered' suffix to the file name
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"sub-{subject}_task-memento_spectral-density_filtered.png",
            ],
            subject,
        )
    else:
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"sub-{subject}_task-memento_spectral-density.png",
            ],
            subject,
        )
    fig = raw.plot_psd()
    fig.savefig(fname)


def plot_noisy_channel_detection(
    auto_scores, subject="test", ch_type="grad", outpath="/tmp/"
):

    # Select the data for specified channel type
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]
    scores = auto_scores["scores_noisy"][ch_subset]
    limits = auto_scores["limits_noisy"][ch_subset]
    bins = auto_scores["bins"]  # The the windows that were evaluated.
    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = pd.DataFrame(
        data=scores,
        columns=pd.Index(bin_labels, name="Time (s)"),
        index=pd.Index(ch_names, name="Channel"),
    )

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(
        f"Automated noisy channel detection: {ch_type}, subject sub-{subject}",
        fontsize=16,
        fontweight="bold",
    )
    sns.heatmap(data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0])
    [
        ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(bins))
    ]
    ax[0].set_title("All Scores", fontweight="bold")

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(
        data=data_to_plot,
        vmin=np.nanmin(limits),  # bads in input data have NaN limits
        cmap="Reds",
        cbar_kws=dict(label="Score"),
        ax=ax[1],
    )
    [
        ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(bins))
    ]
    ax[1].set_title("Scores > Limit", fontweight="bold")

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = _construct_path(
        [
            Path(outpath),
            f"sub-{subject}",
            "meg",
            f"noise_detection_sub-{subject}_{ch_type}.png",
        ],
        subject,
    )
    fig.savefig(fname)


def eventreader(raw, subject, event_dict, outputdir="/tmp/"):
    """
    the Triggers 32790 32792 seem spurious. TODO.
    :param raw:
    :return:
    """
    # for some reason, events only start at sample 628416 (sub 4)
    events = mne.find_events(
        raw,
        min_duration=0.002,  # ignores spurious events
        uint_cast=True,  # workaround an Elekta acquisition bug that causes negative values
        # initial_event=True # unsure - the first on is 32772
    )

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
        ],
        subject,
    )
    fig.savefig(str(fpath))
    return events


def epoch_data(
    raw,
    events,
    event_dict,
    subject,
    conditionname,
    sensor_picks,
    pick_description,
    figdir,
    reject_criteria,
    tmin,
    tmax,
    reject_bad_epochs=True,
    autoreject=False,
):
    """
    Create epochs from specified events.
    :param tmin: int, start time before event. Defaults to -0.2 in MNE 0.23dev
    :param tmax: int, end time after event. Defaults to 0.5 in MNE 0.23dev
    :param sensor_picks: list, all sensors that should be plotted and used for
    epoching
    :param pick_description: str, a short description (no spaces) of the picks,
    e.g., 'occipital' or 'motor'.
    :param figdir: str, Path to where diagnostic plots should be saved.

    TODO: we could include a baseline correction
    TODO: figure out projections -> don't use if you can use SSS
    TODO: autoreject requires picking only MEG channels in epoching
    """

    epoch_params = {
        "raw": raw,
        "events": events,
        "event_id": event_dict,
        "tmin": tmin,
        "tmax": tmax,
        "preload": True,
        "on_missing": "warn",
        "verbose": True,
        "picks": sensor_picks,
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
        # if we want to perform autorejection of epochs using the autoreject tool
        for condition in conditionname:
            # do this for all relevant conditions
            epochs = autoreject_bad_epochs(
                epochs=epochs, picks=sensor_picks, key=condition
            )

    for condition in conditionname:
        _plot_epochs(
            raw,
            epochs=epochs,
            subject=subject,
            key=condition,
            figdir=figdir,
            picks=sensor_picks,
            pick_description=pick_description,
        )
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
    # subselect the required condition. For example visuals = epochs['visualfirst']
    wanted_epochs = epochs[key]
    average = wanted_epochs.average()
    # Some general plots over  all channels
    # make joint plot of topographies and sensors
    figpath_grad = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_avg-epoch_cond-{key}_joint-grad.png",
        ],
        subject,
    )
    figpath_mag = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_avg-epoch_cond-{key}_joint-mag.png",
        ],
        subject,
    )
    fig = average.plot_joint()
    fig1 = fig[0]
    fig2 = fig[1]
    fig1.savefig(figpath_grad)
    fig2.savefig(figpath_grad)
    # also plot topographies
    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_avg-epoch_cond-{key}_topography.png",
        ],
        subject,
    )
    fig = average.plot_topo()
    fig.savefig(figpath)
    # also save the data as an image
    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_avg-epoch_cond-{key}_image.png",
        ],
        subject
    )
    fig = average.plot_image()
    fig.savefig(figpath)

    if picks:
        # If we want to plot a predefined sensor space, e.g., right parietal or left
        # temporal, load in those lists of sensors
        sensors = _get_channel_subsets(raw)
        if picks in sensors.keys():
            picks = sensors[picks]
            #TODO plot with pick_description

    return


def evoked_visual_potentials(raw,
                             subject,
                             event_dict,
                             figdir="/tmp/"):
    """
    Function to plot visual potentials from the first visual option (left) as a
    sanity check
    """

    events = eventreader(raw=raw, subject=subject, outputdir=figdir,
                         event_dict=event_dict)

    visual_epochs = epoch_data(raw=raw,
                               events=events,
                               event_dict=event_dict,
                               subject=subject,
                               conditionname=['visualfirst'],
                               sensor_picks=None,
                               pick_description=None,
                               figdir=figdir,
                               tmax=0.7,
                               tmin=-0.2,
                               reject_criteria=reject_criteria,
                               reject_bad_epochs=True,
                               autoreject=False)


def autoreject_bad_epochs(epochs, key):
    import autoreject
    import numpy as np

    # these values come straight from the tutorial:
    # http://autoreject.github.io/auto_examples/plot_auto_repair.html
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    #important: Requires epochs with only MEG sensors, selected during epoching!
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
