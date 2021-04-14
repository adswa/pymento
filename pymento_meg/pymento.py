import mne
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
    rawdir, subject, bidsdir, figdir, crosstalk_file, fine_cal_file, behav_dir
):
    """
    Transform the original memento MEG data into something structured.
    :return:
    """

    print(
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
    print(
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

    print(
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


def epoch_and_clean_trials(raw, subject, diagdir, bidsdir, datadir, derivdir):
    """
    Chunk the data into epochs starting at the fixation cross at the start of a
    trial, lasting 7 seconds (which should include all trial elements).
    Do automatic artifact detection, rejection and fixing for eyeblinks,
    heartbeat, and high- and low-amplitude artifacts.
    """
    # construct name of the first split
    raw_fname = Path(datadir) / f'sub-{subject}/meg' / \
                f'sub-{subject}_task-memento_proc-sss_meg.fif'
    print(f"Reading in SSS-processed data from subject sub-{subject}. "
          f"Attempting the following path: {raw_fname}")

    # ensure the data is loaded
    raw.load_data()

    # ICA to detect and repair artifacts

    remove_eyeblinks_and_heartbeat(raw=raw,
                                   subject=subject,
                                   figdir=diagdir,
                                   )
    # filter the data to remove high-frequency noise. Minimal high-pass filter
    # based on
    # https://www.sciencedirect.com/science/article/pii/S0165027021000157
    _filter_data(raw, l_freq=0.05, h_freq=40)
    # now, get actual events and epochs
    events, event_dict = get_events(raw)
    # get the actual epochs: chunk the trial into epochs starting from the
    # fixation cross. Do not baseline correct the data.
    epochs = mne.Epochs(raw, events, event_id={'visualfix/fixCross': 10},
                        tmin=0, tmax=7,
                        picks='meg', baseline=(0, 0))
    # TODO: ADD SUBJECT SPECIFIC TRIAL NUMBER TO THE EPOCH! ONLY THIS WAY WE CAN
    # LATER RECOVER WHICH TRIAL PARAMETERS WE'RE LOOKING AT BASED ON THE LOGS AS
    # THE EPOCH REJECTION WILL REMOVE TRIALS
    from pymento_meg.proc.epoch import get_trial_features
    metadata = get_trial_features(bids_path=bidsdir,
                                  subject=subject,
                                  column='trial_no')
    epochs.metadata = metadata
    # use autoreject to repair bad epochs
    epochs.load_data()
    ar = AutoReject()
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
    epochs_clean.save(outpath)
   # visuals = epochs_clean['visualfirst']
   # avg = visuals.average()

   # from pymento_meg.utils import _plot_evoked_fields
   # _plot_evoked_fields(data=avg, subject=subject, figdir=diagdir)
