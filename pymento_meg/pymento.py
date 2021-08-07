import mne
import logging
from pathlib import Path
from pymento_meg.utils import (
    _construct_path,
)
from pymento_meg.orig.restructure import (
    read_data_original,
)
from pymento_meg.proc.preprocess import (
    maxwellfilter,
    _filter_data,
)
from pymento_meg.proc.bids import (
    read_bids_data,
    get_events,
)
from pymento_meg.proc.artifacts import (
    remove_eyeblinks_and_heartbeat,
)
from autoreject import (
    AutoReject,
)


def restructure_to_bids(
        rawdir, subject, bidsdir, figdir, crosstalk_file, fine_cal_file,
        behav_dir
):
    """
    Transform the original memento MEG data into something structured.
    :return:
    """

    logging.info(
        f"Starting to restructure original memento data into BIDS for "
        f"subject sub-{subject}."
    )

    raw = read_data_original(
        directory=rawdir,
        subject=subject,
        savetonewdir=True,
        bidsdir=bidsdir,
        figdir=figdir,
        crosstalk_file=crosstalk_file,
        fine_cal_file=fine_cal_file,
        preprocessing="Raw",
        behav_dir=behav_dir,
    )


def signal_space_separation(bidspath, subject, figdir, derived_path):
    """
    Reads in the raw data from a bids structured directory, applies a basic
    signal space separation with motion correction, and saves the result in a
    derivatives BIDS directory
    :param bidspath:
    :param subject: str, subject identifier, e.g., '001'
    :param figdir: str, path to a diagnostics directory to save figures into
    :param derived_path: str, path to where a derivatives dataset with sss data
    shall be saved
    :return:
    """
    logging.info(
        f"Starting to read in raw memento data from BIDS directory for"
        f"subject sub-{subject}."
    )

    raw, bids_path = read_bids_data(
        bids_root=bidspath,
        subject=subject,
        datatype="meg",
        task="memento",
        suffix="meg",
    )
    # Events are now Annotations, also get them as events
    events = get_events(raw)

    fine_cal_file = bids_path.meg_calibration_fpath
    crosstalk_file = bids_path.meg_crosstalk_fpath

    logging.info(
        f"Starting signal space separation with motion correction "
        f"for subject sub{subject}."
    )

    raw_sss = maxwellfilter(
        raw=raw,
        crosstalk_file=crosstalk_file,
        fine_cal_file=fine_cal_file,
        subject=subject,
        headpos_file=None,
        compute_motion_params=True,
        figdir=figdir,
        outdir=derived_path,
        filtering=False,
        filter_args=None,
    )


def epoch_and_clean_trials(subject,
                           diagdir,
                           bidsdir,
                           datadir,
                           derivdir,
                           eventid={'visualfix/fixCross': 10}):
    """
    Chunk the data into epochs starting at the eventid specified per trial,
    lasting 7 seconds (which should include all trial elements).
    Do automatic artifact detection, rejection and fixing for eyeblinks,
    heartbeat, and high- and low-amplitude artifacts.
    :param subject: str, subject identifier. takes the form '001'
    :param diagdir: str, path to a directory where diagnostic plots can be saved
    :param bidsdir: str, path to a directory with BIDS data. Needed to load
    event logs from the experiment
    :param datadir: str, path to a directory with SSS-processed data
    :param derivdir: str, path to a directory where cleaned epochs can be saved
    :param eventid: dict, the event to start an Epoch from
    """
    # construct name of the first split
    raw_fname = Path(datadir) / f'sub-{subject}/meg' / \
                f'sub-{subject}_task-memento_proc-sss_meg.fif'
    logging.info(f"Reading in SSS-processed data from subject sub-{subject}. "
          f"Attempting the following path: {raw_fname}")
    raw = mne.io.read_raw_fif(raw_fname)
    events, event_dict = get_events(raw)
    # filter the data to remove high-frequency noise. Minimal high-pass filter
    # based on
    # https://www.sciencedirect.com/science/article/pii/S0165027021000157
    # ensure the data is loaded prior to filtering
    raw.load_data()
    # high-pass doesn't make sense, raw data has 0.1Hz high-pass filter already!
    _filter_data(raw, h_freq=40)
    # ICA to detect and repair artifacts
    logging.info('Removing eyeblink and heartbeat artifacts')
    remove_eyeblinks_and_heartbeat(raw=raw,
                                   subject=subject,
                                   figdir=diagdir,
                                   events=events,
                                   eventid=eventid
                                   )
    # get the actual epochs: chunk the trial into epochs starting from the
    # fixation cross. Do not baseline correct the data.
    logging.info('Creating epochs')
    epochs = mne.Epochs(raw, events, event_id=eventid,
                        tmin=0, tmax=7,
                        picks='meg', baseline=None)
    # ADD SUBJECT SPECIFIC TRIAL NUMBER TO THE EPOCH! ONLY THIS WAY WE CAN
    # LATER RECOVER WHICH TRIAL PARAMETERS WE'RE LOOKING AT BASED ON THE LOGS AS
    # THE EPOCH REJECTION WILL REMOVE TRIALS
    logging.info("Retrieving trial metadata.")
    from pymento_meg.proc.epoch import get_trial_features
    metadata = get_trial_features(bids_path=bidsdir,
                                  subject=subject,
                                  column='trial_no')
    # transform to integers
    metadata = metadata.astype(int)
    # this does not work if we start at fixation cross for subject 5, because 1
    # fixation cross trigger is missing from the data, and it becomes impossible
    # to associate the trial metadata to the correct trials in the data
    epochs.metadata = metadata
    epochs.load_data()
    # downsample the data to 100Hz
    logging.info('Resampling epoched data down to 100 Hz')
    epochs.resample(sfreq=100, verbose=True)
    # use autoreject to repair bad epochs
    ar = AutoReject(random_state=42)
    epochs_clean = ar.fit_transform(epochs)
    # save the cleaned, epoched data to disk.
    outpath = _construct_path(
        [
            Path(derivdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_cleaned_epo.fif",
        ]
    )
    logging.info(f"Saving cleaned, epoched data to {outpath}")
    epochs_clean.save(outpath, overwrite=True)
    # visualize the bad sensors for each trial
    fig = ar.get_reject_log(epochs).plot()
    fname = _construct_path(
        [
            Path(diagdir),
            f"sub-{subject}",
            "meg",
            f"epoch-rejectlog_sub-{subject}.png",
        ]
    )
    fig.savefig(fname)


def SRM(subject,
        datadir,
        bidsdir,
        figdir,
        condition='left-right',
        model='srm',
        timespan='fulltrial'):
    """
    Fit an SRM of a certain type with varying amount of features to a condition
    of choice
    :param subject:
    :param datadir:
    :param bidsdir:
    :param figdir:
    :param condition:
    :param model:
    :return:
    """

    if model == 'srm':
        from pymento_meg.srm.srm import plot_trial_components_from_srm
        plot_trial_components_from_srm(subject=subject,
                                          datadir=datadir,
                                          bidsdir=bidsdir,
                                          figdir=figdir,
                                          condition=condition,
                                          timespan=timespan)
