import mne
import os
import logging

from pathlib import Path
from pymento_meg.utils import (
    _construct_path,
)
from pymento_meg.proc.bids import (
    save_derivatives_to_bids_dir,
)
from pymento_meg.viz.plots import (
    plot_psd,
    plot_noisy_channel_detection,
)

preconditioned = False
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def motion_estimation(subject, raw, figdir="/tmp/"):
    """
    Calculate head positions from HPI coils as a prerequisite for movement
    correction.
    :param subject: str, subject identifier; used for writing file names &
    logging
    :param raw: Raw data object
    :param figdir: str, path to directory for diagnostic plots
    """
    # Calculate head motion parameters to remove them during maxwell filtering
    # First, extract HPI coil amplitudes to
    logging.info(f"Extracting HPI coil amplitudes for subject sub-{subject}")
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    # compute time-varying HPI coil locations from amplitudes
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    logging.info(f"Computing head positions for subject sub-{subject}")
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
    # For now, DON'T save headpositions. It is unclear in which BIDS directory.
    # TODO: Figure out whether we want to save them.
    # save head positions
    # outpath = _construct_path(
    #    [
    #        Path(head_pos_outdir),
    #        f"sub-{subject}",
    #        "meg",
    #        f"sub-{subject}_task-memento_headshape.pos",
    #    ]
    # )
    # logging.info(f"Saving head positions as {outpath}")
    # mne.chpi.write_head_pos(outpath, head_pos)

    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_headmovement.png",
        ]
    )
    fig = mne.viz.plot_head_positions(head_pos, mode="traces")
    fig.savefig(figpath)
    figpath = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_headmovement_scaled.png",
        ]
    )
    fig = mne.viz.plot_head_positions(
        head_pos, mode="traces", destination=raw.info["dev_head_t"], info=raw.info
    )
    fig.savefig(figpath)
    return head_pos


def maxwellfilter(
    raw,
    crosstalk_file,
    fine_cal_file,
    subject,
    headpos_file=None,
    compute_motion_params=True,
    figdir="/tmp/",
    outdir="/tmp/",
    filtering=False,
    filter_args=None,
):
    """

    :param raw: Raw data to apply SSS on
    :param crosstalk_file: crosstalk compensation file from the Elekta system to
     reduce interference between gradiometers and magnetometers
    :param fine_cal_file: site-specific sensor orientation and calibration
    :param subject: str, subject identifier, takes the form '001'
    :param headpos_file: str, path; If existing, read in head positions from a file
    :param compute_motion_params: Boolean, whether to perform motion correction
    :param figdir: str, path to directory to save figures in
    :param outdir: str, path to save bids compliant sss-corrected data in
    (derivatives directory)
    :param filtering: if True, a filter function is ran on the data after SSS.
    By default, it is a 40Hz low-pass filter.
    :param filter_args: dict; if filtering is True, initializes a filter with the
    arguments provided

    :return:
    """
    from mne.preprocessing import find_bad_channels_maxwell

    if not compute_motion_params:
        if not headpos_file or not os.path.exists(headpos_file):
            logging.info(
                f"Could not find or read head position files under the supplied"
                f"path: {headpos_file}. Recalculating from scratch."
            )
            head_pos = motion_estimation(subject, raw, figdir)
        else:
            logging.info(
                f"Reading in head positions for subject sub-{subject} "
                f"from {headpos_file}."
            )
            head_pos = mne.chpi.read_head_pos(headpos_file)

    else:
        logging.info(f"Starting motion estimation for subject sub-{subject}.")
        head_pos = motion_estimation(subject, raw, figdir)

    raw.info["bads"] = []
    raw_check = raw.copy()

    preconditioned = False  # TODO handle this here atm. Needs to become global.
    if preconditioned:
        # preconditioned is a global variable that is set to True if some form
        # of filtering (CHPI and line noise removal or general filtering) has
        # been applied.
        # the data has been filtered, and we can pass h_freq=None
        logging.info("Performing bad channel detection without filtering")
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
    logging.info(
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
    logging.info(
        f"Signal Space Separation with movement compensation "
        f"starting for subject sub-{subject}"
    )
    raw_sss = mne.preprocessing.maxwell_filter(
        raw,
        cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        head_pos=head_pos,
        verbose=True,
    )
    # save processed files into their own BIDS directory
    save_derivatives_to_bids_dir(raw_sss=raw_sss, subject=subject, bidsdir=outdir, figdir=figdir)

    if filtering:
        logging.info(
            f"Filtering raw SSS data for subject {subject}. The following "
            f"additional parameters were passed: {filter_args}"
        )
        raw_sss_filtered = raw_sss.copy()
        raw_sss_filtered = _filter_data(raw_sss_filtered, **filter_args)
        # TODO: Downsample
        plot_psd(raw_sss_filtered, subject, figdir, filtering)
        # TODO: save file
        return raw_sss_filtered

    fig = plot_psd(raw_sss, subject, figdir, filtering)
    return raw_sss


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
    logging.info("Loading data for CHPI and line noise filtering")
    raw.load_data()
    logging.info("Applying CHPI and line noise filter")
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
    Filter raw data. This is an exact invocation of the filter function of
    mne 0.23 dev.
    It uses all defaults of this version to ensure future updates to the
    defaults will not break the analysis result reproducibility.
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


def _downsample(raw, frequency):
    """
    Downsample data using MNE's built-in resample function
    """
    raw_downsampled = raw.copy().resample(sfreq=frequency, verbose=True)
    return raw_downsampled
